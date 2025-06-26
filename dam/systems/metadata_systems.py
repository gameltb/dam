"""
This module defines systems related to metadata extraction for assets.

Systems in this module are responsible for processing entities (typically those
marked with `NeedsMetadataExtractionComponent`) to extract and store detailed
metadata such as dimensions, duration, frame counts, audio properties, etc.,
using tools like the Hachoir library.
"""
import asyncio
import logging  # Changed back to logging
from pathlib import Path
from typing import List, Annotated  # Import Annotated and List for the type hint

from dam.core.components_markers import NeedsMetadataExtractionComponent
from dam.core.stages import SystemStage
from dam.core.system_params import (
    CurrentWorldConfig,
    WorldSession,
)
from dam.core.systems import system  # Assuming @system decorator is in dam.core.systems
from dam.models import (
    AudioPropertiesComponent,
    Entity,
    FileLocationComponent,  # Import FileLocationComponent
    FilePropertiesComponent,
    FramePropertiesComponent,
    ImageDimensionsComponent,
)
from dam.services import ecs_service  # For get_component, add_component_to_entity, remove_component

# Hachoir for metadata extraction (can be kept here or moved to a utility if widely used)
try:
    from hachoir.core import config as HachoirConfig
    from hachoir.metadata import extractMetadata
    from hachoir.parser import createParser

    HachoirConfig.quiet = True
except ImportError:
    createParser = None
    extractMetadata = None

logger = logging.getLogger(__name__)


# Helper functions for Hachoir metadata (kept local to this system module)
def _has_hachoir_metadata(md, key: str) -> bool:
    """Safely checks if Hachoir metadata object has a given key."""
    try:
        return md.has(key)
    except (KeyError, ValueError): # Hachoir can sometimes raise ValueError for invalid keys/access
        return False


def _get_hachoir_metadata(md, key: str, default=None) -> Any:
    """Safely gets a value from Hachoir metadata object for a given key."""
    try:
        if md.has(key): # Check first to avoid potential exceptions from get() on some metadata items
            return md.get(key)
    except (KeyError, ValueError): # Hachoir can sometimes raise ValueError
        pass
    return default


def _extract_metadata_with_hachoir_sync(filepath_on_disk: Path) -> Any | None:  # Changed to def
    """
    Synchronous part of Hachoir metadata extraction.
    This function is intended to be run in a separate thread using `asyncio.to_thread`
    to avoid blocking the main asyncio event loop, as Hachoir operations can be I/O bound.

    Args:
        filepath_on_disk: The path to the file from which to extract metadata.

    Returns:
        A Hachoir metadata object if successful, otherwise None.
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


# from typing import Annotated # Moved to top


@system(stage=SystemStage.METADATA_EXTRACTION)
async def extract_metadata_on_asset_ingested(
    session: WorldSession,  # Injected SQLAlchemy Session
    world_config: CurrentWorldConfig,  # Injected WorldConfig
    entities_to_process: Annotated[
        List[Entity], "MarkedEntityList", NeedsMetadataExtractionComponent
    ],  # Corrected usage
    # file_ops: Resource[FileOperationsResource] # If direct file ops are needed beyond path resolution
):
    if not createParser or not extractMetadata:
        logger.warning("Hachoir library not installed. Skipping metadata extraction system.")
        # Remove marker components even if Hachoir is not available to prevent re-processing attempts
        for entity in entities_to_process:
            marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
            if marker:
                ecs_service.remove_component(session, marker, flush=False)
        if entities_to_process:  # Only flush if there were components to remove
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
            # Remove marker and continue
            marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
            if marker:
                ecs_service.remove_component(session, marker, flush=False)
            continue

        mime_type = file_props.mime_type

        # Determine the filepath_on_disk
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

        # Run Hachoir parsing in a separate thread
        metadata = await asyncio.to_thread(_extract_metadata_with_hachoir_sync, filepath_on_disk)

        if not metadata:
            logger.info(f"No metadata extracted by Hachoir for {filepath_on_disk} (Entity ID {entity.id})")
        else:
            logger.debug(f"Hachoir metadata keys for {filepath_on_disk}: {[item for item in metadata]}")
            # --- Populate ImageDimensionsComponent ---
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

            # --- Heuristics and other components (Audio, Frame) ---
            has_duration = _has_hachoir_metadata(metadata, "duration")
            has_width = _has_hachoir_metadata(metadata, "width")
            has_frame_rate = _has_hachoir_metadata(metadata, "frame_rate")
            has_audio_codec = _has_hachoir_metadata(metadata, "audio_codec")
            has_sample_rate = _has_hachoir_metadata(metadata, "sample_rate")

            is_video_heuristic = mime_type.startswith("video/") or (has_duration and (has_width or has_frame_rate))

            is_audio_file_heuristic = False
            if mime_type == "image/gif":  # GIFs are not audio files
                is_audio_file_heuristic = False
            elif mime_type.startswith("audio/"):
                is_audio_file_heuristic = True
            elif has_audio_codec and not is_video_heuristic:
                is_audio_file_heuristic = True
            elif not mime_type.startswith("image/") and not is_video_heuristic and has_duration and has_sample_rate:
                is_audio_file_heuristic = True

            if is_audio_file_heuristic:
                if not ecs_service.get_components(session, entity.id, AudioPropertiesComponent):
                    # ... (populate and add AudioPropertiesComponent, same as before) ...
                    # This part needs to be filled in with the original logic for AudioProperties
                    audio_comp = AudioPropertiesComponent(entity_id=entity.id, entity=entity)  # Ensure entity is passed
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
                    # ... (populate and add FramePropertiesComponent for video, same as before) ...
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
                        # ... (populate and add AudioPropertiesComponent for video's audio, same as before) ...
                        video_audio_comp = AudioPropertiesComponent(entity_id=entity.id, entity=entity)
                        video_duration_audio = _get_hachoir_metadata(
                            metadata, "duration"
                        )  # Use a different var name to avoid conflict
                        if video_duration_audio:
                            video_audio_comp.duration_seconds = video_duration_audio.total_seconds()
                        video_audio_comp.codec_name = _get_hachoir_metadata(metadata, "audio_codec")
                        video_audio_comp.sample_rate_hz = _get_hachoir_metadata(metadata, "sample_rate")
                        video_audio_comp.channels = _get_hachoir_metadata(metadata, "nb_channel")
                        ecs_service.add_component_to_entity(session, entity.id, video_audio_comp, flush=False)
                        logger.info(f"Added AudioPropertiesComponent for video's audio stream, Entity ID {entity.id}")

            if mime_type == "image/gif":
                if not ecs_service.get_components(session, entity.id, FramePropertiesComponent):
                    # ... (populate and add FramePropertiesComponent for GIF, same as before) ...
                    frame_comp = FramePropertiesComponent(entity_id=entity.id, entity=entity)
                    nb_frames_gif = _get_hachoir_metadata(metadata, "nb_frames") or _get_hachoir_metadata(
                        metadata, "frame_count"
                    )  # Use a different var name
                    frame_comp.frame_count = nb_frames_gif
                    duration_gif = _get_hachoir_metadata(metadata, "duration")  # Use a different var name
                    if nb_frames_gif and nb_frames_gif > 1 and duration_gif:
                        duration_sec = duration_gif.total_seconds()
                        frame_comp.animation_duration_seconds = duration_sec
                        if duration_sec > 0:
                            frame_comp.nominal_frame_rate = nb_frames_gif / duration_sec
                    ecs_service.add_component_to_entity(session, entity.id, frame_comp, flush=False)
                    logger.info(f"Added FramePropertiesComponent for animated GIF Entity ID {entity.id}")

        # Remove the marker component after processing
        marker = ecs_service.get_component(session, entity.id, NeedsMetadataExtractionComponent)
        if marker:
            ecs_service.remove_component(session, marker, flush=False)  # Batch flush at end of stage
            logger.debug(f"Removed NeedsMetadataExtractionComponent from Entity ID {entity.id}")

    # Session flush/commit will be handled by the WorldScheduler after the stage execution
    # or after each system if configured that way. For now, assuming after stage.
    # The remove_component calls above use flush=False.
    # The WorldScheduler's current implementation does a flush after removing markers for a system,
    # and then a commit at the end of the stage.
    logger.info("MetadataExtractionSystem finished processing entities.")
