"""
This module defines systems related to metadata extraction for assets.

Systems in this module are responsible for processing entities (typically those
marked with `NeedsMetadataExtractionComponent`) to extract and store detailed
metadata such as dimensions, duration, frame counts, audio properties, etc.,
using tools like the Hachoir library.
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Annotated, Any # Added Any

from dam.core.components_markers import NeedsMetadataExtractionComponent
from dam.core.stages import SystemStage
from dam.core.system_params import (
    CurrentWorldConfig,
    WorldSession,
)
from dam.core.systems import system
from dam.models import (
    AudioPropertiesComponent,
    Entity,
    FileLocationComponent,
    FilePropertiesComponent,
    FramePropertiesComponent,
    ImageDimensionsComponent,
)
from dam.services import ecs_service

try:
    from hachoir.core import config as HachoirConfig
    from hachoir.metadata import extractMetadata
    from hachoir.parser import createParser

    HachoirConfig.quiet = True
except ImportError:
    createParser = None
    extractMetadata = None

logger = logging.getLogger(__name__)


def _has_hachoir_metadata(md, key: str) -> bool:
    """Safely checks if Hachoir metadata object has a given key."""
    try:
        return md.has(key)
    except (KeyError, ValueError):
        return False


def _get_hachoir_metadata(md, key: str, default=None) -> Any:
    """Safely gets a value from Hachoir metadata object for a given key."""
    try:
        if md.has(key):
            return md.get(key)
    except (KeyError, ValueError):
        pass
    return default


async def _extract_metadata_with_hachoir_sync(filepath_on_disk: Path) -> Any | None:
    """
    Synchronous part of Hachoir metadata extraction.
    Run in a separate thread using `asyncio.to_thread`.
    """
    if not createParser or not extractMetadata:
        logger.info("Hachoir library not available. Cannot perform sync extraction.")
        return None

    parser = createParser(str(filepath_on_disk))
    if not parser:
        logger.warning(f"Hachoir could not create a parser for file: {filepath_on_disk}")
        return None

    with parser:
        try:
            metadata = extractMetadata(parser)
            return metadata
        except Exception as e:
            logger.error(f"Hachoir failed to extract metadata for {filepath_on_disk}: {e}", exc_info=True)
            return None


@system(stage=SystemStage.METADATA_EXTRACTION)
async def extract_metadata_on_asset_ingested(
    session: WorldSession,
    world_config: CurrentWorldConfig,
    entities_to_process: Annotated[
        List[Entity], "MarkedEntityList", NeedsMetadataExtractionComponent
    ],
):
    if not createParser or not extractMetadata:
        logger.warning("Hachoir library not installed. Skipping metadata extraction system.")
        for entity in entities_to_process:
            marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
            if marker:
                ecs_service.remove_component(session, marker, flush=False)
        if entities_to_process:
            session.flush()
        return

    if not entities_to_process:
        logger.debug("No entities marked for metadata extraction in this run.")
        return

    logger.info(
        f"MetadataExtractionSystem running for {len(entities_to_process)} entities in world '{world_config.DATABASE_URL}'."
    )

    for entity in entities_to_process:
        logger.debug(f"Processing entity ID {entity.id} for metadata extraction.")

        file_props = ecs_service.get_component(session, entity.id, FilePropertiesComponent)
        if not file_props:
            logger.warning(f"No FilePropertiesComponent found for Entity ID {entity.id}. Cannot extract metadata.")
            marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
            if marker:
                ecs_service.remove_component(session, marker, flush=False)
            continue

        mime_type = file_props.mime_type
        all_locations = ecs_service.get_components(session, entity.id, FileLocationComponent)
        if not all_locations:
            logger.warning(f"No FileLocationComponent found for Entity ID {entity.id}. Cannot extract metadata.")
            marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
            if marker:
                ecs_service.remove_component(session, marker, flush=False)
            continue

        filepath_on_disk: Path | None = None
        ref_loc = next((loc for loc in all_locations if loc.storage_type == "local_reference"), None)
        cas_loc = next((loc for loc in all_locations if loc.storage_type == "local_cas"), None)

        if ref_loc:
            filepath_on_disk = Path(ref_loc.physical_path_or_key)
        elif cas_loc:
            base_storage_path = Path(world_config.ASSET_STORAGE_PATH)
            filepath_on_disk = base_storage_path / cas_loc.physical_path_or_key

        if not filepath_on_disk or not await asyncio.to_thread(filepath_on_disk.exists):
            logger.error(
                f"Filepath '{filepath_on_disk}' for Entity ID {entity.id} does not exist. Cannot extract metadata."
            )
            marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
            if marker:
                ecs_service.remove_component(session, marker, flush=False)
            continue

        logger.info(f"Extracting metadata from {filepath_on_disk} for Entity ID {entity.id} (MIME: {mime_type})")
        metadata = await asyncio.to_thread(_extract_metadata_with_hachoir_sync, filepath_on_disk)

        if not metadata:
            logger.info(f"No metadata extracted by Hachoir for {filepath_on_disk} (Entity ID {entity.id})")
        else:
            logger.debug(f"Hachoir metadata keys for {filepath_on_disk}: {[item for item in metadata]}")
            if mime_type.startswith("image/") or mime_type.startswith("video/"):
                if not ecs_service.get_components(session, entity.id, ImageDimensionsComponent):
                    width = _get_hachoir_metadata(metadata, "width")
                    height = _get_hachoir_metadata(metadata, "height")
                    if width is not None and height is not None:
                        dim_comp = ImageDimensionsComponent(
                            entity_id=entity.id, entity=entity, width_pixels=width, height_pixels=height
                        )
                        ecs_service.add_component_to_entity(session, entity.id, dim_comp, flush=False)
                        logger.info(f"Added ImageDimensionsComponent ({width}x{height}) for Entity ID {entity.id}")
                    else:
                        logger.warning(
                            f"Could not extract width/height for visual media Entity ID {entity.id} (MIME: {mime_type})"
                        )

            has_duration = _has_hachoir_metadata(metadata, "duration")
            has_width = _has_hachoir_metadata(metadata, "width")
            has_frame_rate = _has_hachoir_metadata(metadata, "frame_rate")
            has_audio_codec = _has_hachoir_metadata(metadata, "audio_codec")
            has_sample_rate = _has_hachoir_metadata(metadata, "sample_rate")

            is_video_heuristic = mime_type.startswith("video/") or (has_duration and (has_width or has_frame_rate))
            is_audio_file_heuristic = False
            if mime_type == "image/gif":
                is_audio_file_heuristic = False
            elif mime_type.startswith("audio/"):
                is_audio_file_heuristic = True
            elif has_audio_codec and not is_video_heuristic:
                is_audio_file_heuristic = True
            elif not mime_type.startswith("image/") and not is_video_heuristic and has_duration and has_sample_rate:
                is_audio_file_heuristic = True

            if is_audio_file_heuristic:
                if not ecs_service.get_components(session, entity.id, AudioPropertiesComponent):
                    audio_comp = AudioPropertiesComponent(entity_id=entity.id, entity=entity)
                    duration = _get_hachoir_metadata(metadata, "duration")
                    if duration:
                        audio_comp.duration_seconds = duration.total_seconds()
                    audio_codec = _get_hachoir_metadata(metadata, "audio_codec") or _get_hachoir_metadata(
                        metadata, "compression"
                    )
                    audio_comp.codec_name = audio_codec
                    audio_comp.sample_rate_hz = _get_hachoir_metadata(metadata, "sample_rate")
                    audio_comp.channels = _get_hachoir_metadata(metadata, "nb_channel")
                    bit_rate_bps = _get_hachoir_metadata(metadata, "bit_rate")
                    if bit_rate_bps:
                        audio_comp.bit_rate_kbps = bit_rate_bps // 1000
                    ecs_service.add_component_to_entity(session, entity.id, audio_comp, flush=False)
                    logger.info(f"Added AudioPropertiesComponent for standalone audio Entity ID {entity.id}")

            if is_video_heuristic:
                if not ecs_service.get_components(session, entity.id, FramePropertiesComponent):
                    video_frame_comp = FramePropertiesComponent(entity_id=entity.id, entity=entity)
                    video_duration = _get_hachoir_metadata(metadata, "duration")
                    nb_frames = _get_hachoir_metadata(metadata, "nb_frames") or _get_hachoir_metadata(
                        metadata, "frame_count"
                    )
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
                    ecs_service.add_component_to_entity(session, entity.id, video_frame_comp, flush=False)
                    logger.info(f"Added FramePropertiesComponent for video Entity ID {entity.id}")

                if _has_hachoir_metadata(metadata, "audio_codec"):
                    if not ecs_service.get_components(session, entity.id, AudioPropertiesComponent):
                        video_audio_comp = AudioPropertiesComponent(entity_id=entity.id, entity=entity)
                        video_duration_audio = _get_hachoir_metadata(metadata, "duration")
                        if video_duration_audio:
                            video_audio_comp.duration_seconds = video_duration_audio.total_seconds()
                        video_audio_comp.codec_name = _get_hachoir_metadata(metadata, "audio_codec")
                        video_audio_comp.sample_rate_hz = _get_hachoir_metadata(metadata, "sample_rate")
                        video_audio_comp.channels = _get_hachoir_metadata(metadata, "nb_channel")
                        ecs_service.add_component_to_entity(session, entity.id, video_audio_comp, flush=False)
                        logger.info(f"Added AudioPropertiesComponent for video's audio stream, Entity ID {entity.id}")

            if mime_type == "image/gif":
                if not ecs_service.get_components(session, entity.id, FramePropertiesComponent):
                    frame_comp = FramePropertiesComponent(entity_id=entity.id, entity=entity)
                    nb_frames_gif = _get_hachoir_metadata(metadata, "nb_frames") or _get_hachoir_metadata(
                        metadata, "frame_count"
                    )
                    frame_comp.frame_count = nb_frames_gif
                    duration_gif = _get_hachoir_metadata(metadata, "duration")
                    if nb_frames_gif and nb_frames_gif > 1 and duration_gif:
                        duration_sec = duration_gif.total_seconds()
                        frame_comp.animation_duration_seconds = duration_sec
                        if duration_sec > 0:
                            frame_comp.nominal_frame_rate = nb_frames_gif / duration_sec
                    ecs_service.add_component_to_entity(session, entity.id, frame_comp, flush=False)
                    logger.info(f"Added FramePropertiesComponent for animated GIF Entity ID {entity.id}")

        marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
        if marker:
            ecs_service.remove_component(session, marker, flush=False)
            logger.debug(f"Removed NeedsMetadataExtractionComponent from Entity ID {entity.id}")

    logger.info("MetadataExtractionSystem finished processing entities.")
