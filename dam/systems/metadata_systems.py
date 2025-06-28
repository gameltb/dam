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
from typing import Annotated, Any, List  # Added Dict

from dam.core.components_markers import NeedsMetadataExtractionComponent
from dam.core.config import WorldConfig  # Import WorldConfig directly
from dam.core.stages import SystemStage
from dam.core.system_params import WorldSession  # CurrentWorldConfig removed
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


# This is the internal synchronous parsing function
def _parse_metadata_sync_worker(filepath_str: str) -> Any | None:
    if not createParser or not extractMetadata:
        logger.info("Hachoir library not available. Cannot perform sync extraction.")
        return None

    parser = createParser(filepath_str)
    if not parser:
        logger.warning(f"Hachoir could not create a parser for file: {filepath_str}")
        return None

    with parser:
        try:
            metadata = extractMetadata(parser)
            return metadata
        except Exception as e:
            logger.error(f"Hachoir failed to extract metadata for {filepath_str}: {e}", exc_info=True)
            return None


async def _extract_metadata_with_hachoir_sync(filepath_on_disk: Path) -> Any | None:
    """
    Asynchronously calls the synchronous Hachoir metadata extraction.
    """
    return await asyncio.to_thread(_parse_metadata_sync_worker, str(filepath_on_disk))


@system(stage=SystemStage.METADATA_EXTRACTION)
async def extract_metadata_on_asset_ingested(
    session: WorldSession,
    world_config: WorldConfig,  # Changed from CurrentWorldConfig
    entities_to_process: Annotated[List[Entity], "MarkedEntityList", NeedsMetadataExtractionComponent],
):
    if not createParser or not extractMetadata:
        logger.warning("Hachoir library not installed. Skipping metadata extraction system.")
        for entity in entities_to_process:  # Clean up markers if system can't run
            marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
            if marker:
                ecs_service.remove_component(session, marker, flush=False)
        if entities_to_process:  # Only flush if there were entities to potentially remove markers from
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

        mime_type = (
            file_props.mime_type if file_props else "application/octet-stream"
        )  # Default if fpc is None (though checked above)

        all_locations = ecs_service.get_components(session, entity.id, FileLocationComponent)
        if not all_locations:
            logger.warning(f"No FileLocationComponent found for Entity ID {entity.id}. Cannot extract metadata.")
            marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
            if marker:
                ecs_service.remove_component(session, marker, flush=False)
            continue

        filepath_on_disk: Path | None = None
        # Prioritize CAS location if available, then reference
        cas_loc = next((loc for loc in all_locations if loc.storage_type == "local_cas"), None)
        ref_loc = next((loc for loc in all_locations if loc.storage_type == "local_reference"), None)

        if cas_loc:
            base_storage_path = Path(world_config.ASSET_STORAGE_PATH)
            filepath_on_disk = base_storage_path / cas_loc.physical_path_or_key
        elif ref_loc:
            filepath_on_disk = Path(ref_loc.physical_path_or_key)

        if not filepath_on_disk or not await asyncio.to_thread(filepath_on_disk.exists):
            logger.error(
                f"Filepath '{filepath_on_disk}' for Entity ID {entity.id} does not exist or could not be determined. Cannot extract metadata."
            )
            marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
            if marker:
                ecs_service.remove_component(session, marker, flush=False)
            continue

        logger.info(f"Extracting metadata from {filepath_on_disk} for Entity ID {entity.id} (MIME: {mime_type})")

        # Correctly await the async function that internally uses to_thread
        metadata = await _extract_metadata_with_hachoir_sync(filepath_on_disk)

        if not metadata:
            logger.info(f"No metadata extracted by Hachoir for {filepath_on_disk} (Entity ID {entity.id})")
        else:
            keys_to_log = []
            if hasattr(metadata, "keys") and callable(metadata.keys):
                try:
                    keys_to_log = list(metadata.keys())
                except Exception:
                    keys_to_log = ["<error reading Hachoir metadata keys>"]
            # Hachoir metadata objects might not be simple dicts, so direct isinstance(metadata, dict) might be too restrictive.
            # The .has() and .get() methods are safer.
            logger.debug(
                f"Hachoir metadata (type: {type(metadata).__name__}) keys for {filepath_on_disk}: {keys_to_log}"
            )

            if mime_type.startswith("image/") or mime_type.startswith("video/"):
                if not ecs_service.get_components(session, entity.id, ImageDimensionsComponent):
                    width = _get_hachoir_metadata(metadata, "width")
                    height = _get_hachoir_metadata(metadata, "height")
                    if width is not None and height is not None:
                        dim_comp = ImageDimensionsComponent(entity=entity, width_pixels=width, height_pixels=height)
                        ecs_service.add_component_to_entity(session, entity.id, dim_comp, flush=False)
                        logger.info(f"Added ImageDimensionsComponent ({width}x{height}) for Entity ID {entity.id}")
                    else:
                        logger.warning(
                            f"Could not extract width/height for visual media Entity ID {entity.id} (MIME: {mime_type})"
                        )

            # Heuristics for audio/video based on Hachoir fields
            has_duration = _has_hachoir_metadata(metadata, "duration")
            has_width = _has_hachoir_metadata(metadata, "width")  # Often present for video
            has_frame_rate = _has_hachoir_metadata(metadata, "frame_rate")  # Video
            has_audio_codec = _has_hachoir_metadata(metadata, "audio_codec")
            has_sample_rate = _has_hachoir_metadata(metadata, "sample_rate")  # Audio

            is_video_heuristic = (
                mime_type.startswith("video/")
                or (
                    mime_type.startswith("image/")
                    and _get_hachoir_metadata(metadata, "nb_frames", 0) > 1
                    and has_duration
                )
                or (has_duration and (has_width or has_frame_rate) and not mime_type.startswith("audio/"))
            )

            is_audio_file_heuristic = False
            if mime_type.startswith("audio/"):
                is_audio_file_heuristic = True
            elif has_audio_codec and not is_video_heuristic:  # If it has audio codec but isn't identified as video
                is_audio_file_heuristic = True
            # More specific: if not image/video but has duration and audio-specific fields
            elif (
                not mime_type.startswith("image/")
                and not is_video_heuristic
                and has_duration
                and (has_sample_rate or has_audio_codec)
            ):
                is_audio_file_heuristic = True

            if is_audio_file_heuristic:  # Standalone audio file
                if not ecs_service.get_components(session, entity.id, AudioPropertiesComponent):
                    audio_comp = AudioPropertiesComponent(entity=entity)
                    duration = _get_hachoir_metadata(metadata, "duration")
                    if duration:
                        audio_comp.duration_seconds = duration.total_seconds()

                    audio_codec_val = _get_hachoir_metadata(metadata, "audio_codec")
                    if not audio_codec_val and _has_hachoir_metadata(
                        metadata, "compression"
                    ):  # Fallback for some formats
                        audio_codec_val = _get_hachoir_metadata(metadata, "compression")
                    audio_comp.codec_name = audio_codec_val

                    audio_comp.sample_rate_hz = _get_hachoir_metadata(metadata, "sample_rate")
                    audio_comp.channels = _get_hachoir_metadata(metadata, "nb_channel")
                    bit_rate_bps = _get_hachoir_metadata(metadata, "bit_rate")
                    if bit_rate_bps:  # Hachoir often gives total bit_rate for file
                        audio_comp.bit_rate_kbps = bit_rate_bps // 1000

                    ecs_service.add_component_to_entity(session, entity.id, audio_comp, flush=False)
                    logger.info(f"Added AudioPropertiesComponent for standalone audio Entity ID {entity.id}")

            if is_video_heuristic:  # Video file or animated image with frames/duration
                if not ecs_service.get_components(session, entity.id, FramePropertiesComponent):
                    video_frame_comp = FramePropertiesComponent(entity=entity)

                    nb_frames = _get_hachoir_metadata(metadata, "nb_frames") or _get_hachoir_metadata(
                        metadata, "frame_count"
                    )
                    video_frame_comp.frame_count = nb_frames
                    video_frame_comp.nominal_frame_rate = _get_hachoir_metadata(metadata, "frame_rate")

                    video_duration = _get_hachoir_metadata(metadata, "duration")
                    if video_duration:
                        video_frame_comp.animation_duration_seconds = video_duration.total_seconds()

                    # Estimate frame_count if missing but rate and duration are available
                    if (
                        not video_frame_comp.frame_count
                        and video_frame_comp.nominal_frame_rate
                        and video_frame_comp.animation_duration_seconds
                        and video_frame_comp.animation_duration_seconds > 0
                    ):
                        video_frame_comp.frame_count = int(
                            video_frame_comp.nominal_frame_rate * video_frame_comp.animation_duration_seconds
                        )

                    ecs_service.add_component_to_entity(session, entity.id, video_frame_comp, flush=False)
                    logger.info(f"Added FramePropertiesComponent for video/animated Entity ID {entity.id}")

                # If video, check for embedded audio stream
                if has_audio_codec and not ecs_service.get_components(session, entity.id, AudioPropertiesComponent):
                    video_audio_comp = AudioPropertiesComponent(entity=entity)
                    video_duration_audio = _get_hachoir_metadata(metadata, "duration")  # Use overall duration
                    if video_duration_audio:
                        video_audio_comp.duration_seconds = video_duration_audio.total_seconds()
                    video_audio_comp.codec_name = _get_hachoir_metadata(metadata, "audio_codec")
                    video_audio_comp.sample_rate_hz = _get_hachoir_metadata(metadata, "sample_rate")
                    video_audio_comp.channels = _get_hachoir_metadata(metadata, "nb_channel")
                    # Bit rate for audio stream might be separate or part of overall; Hachoir often gives overall.
                    # We'll leave bit_rate_kbps empty for embedded audio unless specifically found for the audio stream.
                    ecs_service.add_component_to_entity(session, entity.id, video_audio_comp, flush=False)
                    logger.info(f"Added AudioPropertiesComponent for video's audio stream, Entity ID {entity.id}")

            # Specific handling for animated GIFs (distinct from general video)
            if mime_type == "image/gif":
                if not ecs_service.get_components(
                    session, entity.id, FramePropertiesComponent
                ):  # Check again if not caught by video heuristic
                    frame_comp = FramePropertiesComponent(entity=entity)
                    nb_frames_gif = _get_hachoir_metadata(metadata, "nb_frames") or _get_hachoir_metadata(
                        metadata, "frame_count"
                    )
                    frame_comp.frame_count = nb_frames_gif

                    duration_gif_obj = _get_hachoir_metadata(metadata, "duration")
                    if duration_gif_obj:
                        duration_sec = duration_gif_obj.total_seconds()
                        frame_comp.animation_duration_seconds = duration_sec
                        if nb_frames_gif and nb_frames_gif > 1 and duration_sec > 0:
                            frame_comp.nominal_frame_rate = nb_frames_gif / duration_sec

                    ecs_service.add_component_to_entity(session, entity.id, frame_comp, flush=False)
                    logger.info(f"Added FramePropertiesComponent for animated GIF Entity ID {entity.id}")

        # Clean up the marker component after processing
        marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
        if marker:
            ecs_service.remove_component(session, marker, flush=False)  # Batch flush at end of system if needed
            logger.debug(f"Removed NeedsMetadataExtractionComponent from Entity ID {entity.id}")

    logger.info("MetadataExtractionSystem finished processing entities.")
    # Session flush will be handled by the WorldScheduler after the stage execution.
    # If individual flushes are needed per entity inside the loop (e.g. to release locks sooner),
    # then add session.flush() after each entity's processing, but be mindful of performance.
    # For now, relying on stage-level flush.
