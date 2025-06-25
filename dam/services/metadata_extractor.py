import logging
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from dam.models import (
    AudioPropertiesComponent,
    Entity,
    FramePropertiesComponent,
    ImageDimensionsComponent,
)
from dam.services.ecs_service import add_component_to_entity, get_components

# Hachoir for metadata extraction
try:
    from hachoir.core import config as HachoirConfig
    from hachoir.metadata import extractMetadata
    from hachoir.parser import createParser

    HachoirConfig.quiet = True  # Suppress Hachoir's console output unless it's an error
except ImportError:
    createParser = None
    extractMetadata = None

logger = logging.getLogger(__name__)


def _has_hachoir_metadata(md, key):
    try:
        return md.has(key)
    except (KeyError, ValueError): # Hachoir can raise these if key is truly absent
        return False


def _get_hachoir_metadata(md, key, default=None):
    try:
        if md.has(key):
            return md.get(key)
    except (KeyError, ValueError):
        pass
    return default


def extract_and_add_multimedia_components(
    session: Session,
    entity: Entity,
    filepath_on_disk: Path,
    mime_type: str,
    world_name_for_log: Optional[str] = "current", # For logging context
):
    """
    Extracts multimedia metadata using Hachoir and adds relevant components
    (ImageDimensionsComponent, FramePropertiesComponent, AudioPropertiesComponent)
    to the given entity if they don't already exist.
    """
    if not createParser or not extractMetadata:
        logger.warning(
            f"Hachoir not available. Cannot extract multimedia metadata for Entity ID {entity.id} in world '{world_name_for_log}'."
        )
        return

    parser = createParser(str(filepath_on_disk))
    if not parser:
        logger.warning(f"Hachoir could not create a parser for file: {filepath_on_disk} (Entity ID {entity.id})")
        return

    with parser:
        try:
            metadata = extractMetadata(parser)
        except Exception as e:
            logger.error(f"Hachoir failed to extract metadata for {filepath_on_disk} (Entity ID {entity.id}): {e}", exc_info=True)
            metadata = None

    if not metadata:
        logger.info(f"No metadata extracted by Hachoir for {filepath_on_disk} (Entity ID {entity.id})")
        return

    # --- Populate ImageDimensionsComponent for any visual media ---
    if mime_type.startswith("image/") or mime_type.startswith("video/"):
        if not get_components(session, entity.id, ImageDimensionsComponent): # Check if component already exists
            width = _get_hachoir_metadata(metadata, "width")
            height = _get_hachoir_metadata(metadata, "height")
            if width is not None and height is not None:
                dim_comp = ImageDimensionsComponent(
                    entity_id=entity.id, entity=entity, width_pixels=width, height_pixels=height
                )
                add_component_to_entity(session, entity.id, dim_comp)
                logger.info(f"Added ImageDimensionsComponent ({width}x{height}) for Entity ID {entity.id}")

    # --- Heuristics for content type based on metadata ---
    is_video_heuristic = mime_type.startswith("video/") or (
        _has_hachoir_metadata(metadata, "duration")
        and (_has_hachoir_metadata(metadata, "width") or _has_hachoir_metadata(metadata, "frame_rate"))
    )

    is_audio_file_heuristic = mime_type.startswith("audio/") or \
        (_has_hachoir_metadata(metadata, "audio_codec") and not is_video_heuristic) or \
        (not is_video_heuristic and _has_hachoir_metadata(metadata, "duration") and _has_hachoir_metadata(metadata, "sample_rate"))


    # --- Populate AudioPropertiesComponent for standalone audio files ---
    if is_audio_file_heuristic:
        if not get_components(session, entity.id, AudioPropertiesComponent):
            audio_comp = AudioPropertiesComponent(entity_id=entity.id, entity=entity)
            duration = _get_hachoir_metadata(metadata, "duration")
            if duration:
                audio_comp.duration_seconds = duration.total_seconds()

            audio_codec = _get_hachoir_metadata(metadata, "audio_codec")
            if not audio_codec:
                audio_codec = _get_hachoir_metadata(metadata, "compression")
            audio_comp.codec_name = audio_codec

            audio_comp.sample_rate_hz = _get_hachoir_metadata(metadata, "sample_rate")
            audio_comp.channels = _get_hachoir_metadata(metadata, "nb_channel")
            bit_rate_bps = _get_hachoir_metadata(metadata, "bit_rate")
            if bit_rate_bps:
                audio_comp.bit_rate_kbps = bit_rate_bps // 1000

            add_component_to_entity(session, entity.id, audio_comp)
            logger.info(f"Added AudioPropertiesComponent for standalone audio Entity ID {entity.id}")

    # --- Populate components for video content ---
    if is_video_heuristic:
        # FramePropertiesComponent for video's visual stream
        if not get_components(session, entity.id, FramePropertiesComponent):
            video_frame_comp = FramePropertiesComponent(entity_id=entity.id, entity=entity)
            video_duration = _get_hachoir_metadata(metadata, "duration")

            nb_frames = _get_hachoir_metadata(metadata, "nb_frames") or _get_hachoir_metadata(metadata, "frame_count")
            video_frame_comp.frame_count = nb_frames
            video_frame_comp.nominal_frame_rate = _get_hachoir_metadata(metadata, "frame_rate")

            if video_duration:
                video_frame_comp.animation_duration_seconds = video_duration.total_seconds()

            if (
                not video_frame_comp.frame_count
                and video_frame_comp.nominal_frame_rate
                and video_frame_comp.animation_duration_seconds
            ):
                video_frame_comp.frame_count = int(
                    video_frame_comp.nominal_frame_rate * video_frame_comp.animation_duration_seconds
                )
            add_component_to_entity(session, entity.id, video_frame_comp)
            logger.info(f"Added FramePropertiesComponent for video Entity ID {entity.id}")

        # AudioPropertiesComponent for video's audio stream(s)
        if _has_hachoir_metadata(metadata, "audio_codec"):
            if not get_components(session, entity.id, AudioPropertiesComponent): # Check if already added
                video_audio_comp = AudioPropertiesComponent(entity_id=entity.id, entity=entity)
                video_duration = _get_hachoir_metadata(metadata, "duration")
                if video_duration:
                    video_audio_comp.duration_seconds = video_duration.total_seconds()
                video_audio_comp.codec_name = _get_hachoir_metadata(metadata, "audio_codec")
                video_audio_comp.sample_rate_hz = _get_hachoir_metadata(metadata, "sample_rate")
                video_audio_comp.channels = _get_hachoir_metadata(metadata, "nb_channel")
                add_component_to_entity(session, entity.id, video_audio_comp)
                logger.info(f"Added AudioPropertiesComponent for video's audio stream, Entity ID {entity.id}")

    # --- Populate FramePropertiesComponent for animated images like GIFs ---
    # This is separate from video heuristic as GIFs are images but have frames.
    if mime_type == "image/gif":
        if not get_components(session, entity.id, FramePropertiesComponent): # Check if already added (e.g. by video logic if mime was video/gif)
            frame_comp = FramePropertiesComponent(entity_id=entity.id, entity=entity)
            nb_frames = _get_hachoir_metadata(metadata, "nb_frames") or _get_hachoir_metadata(metadata, "frame_count")
            frame_comp.frame_count = nb_frames

            duration = _get_hachoir_metadata(metadata, "duration")
            if nb_frames and nb_frames > 1 and duration:
                duration_sec = duration.total_seconds()
                frame_comp.animation_duration_seconds = duration_sec
                if duration_sec > 0:
                    frame_comp.nominal_frame_rate = nb_frames / duration_sec

            add_component_to_entity(session, entity.id, frame_comp)
            logger.info(f"Added FramePropertiesComponent for animated GIF Entity ID {entity.id}")
