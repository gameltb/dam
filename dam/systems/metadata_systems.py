"""
This module defines systems related to metadata extraction for assets.

Systems in this module are responsible for processing entities (typically those
marked with `NeedsMetadataExtractionComponent`) to extract and store detailed
metadata such as dimensions, duration, frame counts, audio properties, etc.,
using tools like the Hachoir library and exiftool.
"""

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Any, Dict, List

from dam.core.components_markers import NeedsMetadataExtractionComponent
from dam.core.config import WorldConfig
from dam.core.stages import SystemStage
from dam.core.system_params import WorldSession
from dam.core.systems import system
from dam.models import (
    AudioPropertiesComponent,
    Entity,
    FileLocationComponent,
    FilePropertiesComponent,
    FramePropertiesComponent,
    ImageDimensionsComponent,
)
from dam.models.metadata.exiftool_metadata_component import ExiftoolMetadataComponent
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


async def _run_exiftool_subprocess(filepath: Path) -> Dict[str, Any] | None:
    """
    Runs exiftool on the given filepath and returns the JSON output.
    Returns None if exiftool is not found or if an error occurs.
    """
    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        logger.warning("exiftool command not found in PATH. Skipping exiftool metadata extraction.")
        return None

    command = [exiftool_path, "-json", "-G", str(filepath)]
    try:
        process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(
                f"Exiftool error for {filepath} (return code {process.returncode}): {stderr.decode(errors='ignore')}"
            )
            return None

        try:
            # exiftool outputs a list with a single JSON object
            data = json.loads(stdout)
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            logger.warning(
                f"Exiftool output for {filepath} was not a list with one element: {stdout.decode(errors='ignore')[:200]}"
            )
            return None
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON from exiftool for {filepath}: {e}. Output: {stdout.decode(errors='ignore')[:200]}"
            )
            return None

    except Exception as e:
        logger.error(f"Exception running exiftool for {filepath}: {e}", exc_info=True)
        return None


async def _extract_metadata_with_exiftool_async(filepath_on_disk: Path) -> Dict[str, Any] | None:
    """
    Asynchronously calls exiftool metadata extraction.
    """
    return await _run_exiftool_subprocess(filepath_on_disk)


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
    world_config: WorldConfig,
    entities_to_process: Annotated[List[Entity], "MarkedEntityList", NeedsMetadataExtractionComponent],
):
    if not createParser or not extractMetadata: # Hachoir check
        logger.warning("Hachoir library not installed. Skipping metadata extraction system.")
        for entity in entities_to_process:
            # Attempt to remove marker even if Hachoir is not there
            markers = await ecs_service.get_components(session, entity.id, NeedsMetadataExtractionComponent)
            for marker_to_remove in markers:
                await ecs_service.remove_component(session, marker_to_remove, flush=False)
        if entities_to_process:
            await session.flush()
        return

    if not entities_to_process:
        logger.debug("No entities marked for metadata extraction in this run.")
        return

    logger.info(
        f"MetadataExtractionSystem running for {len(entities_to_process)} entities in world '{world_config.DATABASE_URL}'."
    )

    for entity in entities_to_process:
        original_entity_id_for_log = entity.id # Capture for logging in case entity becomes None
        logger.debug(f"Processing entity ID {original_entity_id_for_log} for metadata extraction.")

        # Workaround for potential duplicate markers:
        # Fetch all markers, process based on the entity, and ensure all markers for this entity are cleared.
        current_markers = await ecs_service.get_components(session, entity.id, NeedsMetadataExtractionComponent)
        if not current_markers:
            logger.info(f"No NeedsMetadataExtractionComponent found for Entity {entity.id}; possibly already processed or removed.")
            continue

        if len(current_markers) > 1:
            logger.warning(
                f"Entity {entity.id} has multiple NeedsMetadataExtractionComponent instances ({len(current_markers)} found). "
                f"This indicates a potential issue with duplicate marker creation. All will be processed for removal."
            )

        # Main processing logic starts here, assuming at least one marker was found.
        # The marker component itself doesn't hold data needed for extraction,
        # its presence on the entity is what matters for it to be in `entities_to_process`.

        file_props = await ecs_service.get_component(session, entity.id, FilePropertiesComponent)
        if not file_props:
            logger.warning(f"No FilePropertiesComponent found for Entity ID {entity.id}. Cannot extract metadata.")
            # Still try to remove markers
            for m_to_del in current_markers: # Iterate over the fetched list
                await ecs_service.remove_component(session, m_to_del, flush=False)
            continue

        mime_type = file_props.mime_type if file_props else "application/octet-stream"
        all_locations = await ecs_service.get_components(session, entity.id, FileLocationComponent)
        if not all_locations:
            logger.warning(f"No FileLocationComponent found for Entity ID {entity.id}. Cannot extract metadata.")
            for m_to_del in current_markers: # Iterate over the fetched list
                await ecs_service.remove_component(session, m_to_del, flush=False)
            continue

        filepath_on_disk: Path | None = None
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
            for m_to_del in current_markers: # Iterate over the fetched list
                await ecs_service.remove_component(session, m_to_del, flush=False)
            continue

        logger.info(f"Extracting metadata from {filepath_on_disk} for Entity ID {entity.id} (MIME: {mime_type})")
        hachoir_metadata = await _extract_metadata_with_hachoir_sync(filepath_on_disk)

        if not hachoir_metadata:
            logger.info(f"No metadata extracted by Hachoir for {filepath_on_disk} (Entity ID {entity.id})")
        else:
            keys_to_log = []
            if hasattr(hachoir_metadata, "keys") and callable(hachoir_metadata.keys):
                try:
                    keys_to_log = list(hachoir_metadata.keys())
                except Exception:
                    keys_to_log = ["<error reading Hachoir metadata keys>"]
            logger.debug(
                f"Hachoir metadata (type: {type(hachoir_metadata).__name__}) keys for {filepath_on_disk}: {keys_to_log}"
            )

            if mime_type.startswith("image/") or mime_type.startswith("video/"):
                if not await ecs_service.get_components(session, entity.id, ImageDimensionsComponent):
                    width = _get_hachoir_metadata(hachoir_metadata, "width")
                    height = _get_hachoir_metadata(hachoir_metadata, "height")
                    if width is not None and height is not None:
                        dim_comp = ImageDimensionsComponent(width_pixels=width, height_pixels=height)
                        await ecs_service.add_component_to_entity(session, entity.id, dim_comp, flush=False)
                        logger.info(f"Added ImageDimensionsComponent ({width}x{height}) for Entity ID {entity.id}")
                    else:
                        logger.warning(
                            f"Could not extract width/height for visual media Entity ID {entity.id} (MIME: {mime_type})"
                        )
            # (The rest of the Hachoir metadata processing logic from the original file would go here)
            # This includes is_video_heuristic, is_audio_file_heuristic, and adding relevant components.
            # For brevity in this diff, it's omitted but assumed to be the same as the original.
            has_duration = _has_hachoir_metadata(hachoir_metadata, "duration")
            has_width = _has_hachoir_metadata(hachoir_metadata, "width")
            has_frame_rate = _has_hachoir_metadata(hachoir_metadata, "frame_rate")
            has_audio_codec = _has_hachoir_metadata(hachoir_metadata, "audio_codec")
            has_sample_rate = _has_hachoir_metadata(hachoir_metadata, "sample_rate")

            is_video_heuristic = (
                mime_type.startswith("video/")
                or (
                    mime_type.startswith("image/")
                    and _get_hachoir_metadata(hachoir_metadata, "nb_frames", 0) > 1
                    and has_duration
                )
                or (has_duration and (has_width or has_frame_rate) and not mime_type.startswith("audio/"))
            )

            is_audio_file_heuristic = False
            if mime_type.startswith("audio/"):
                is_audio_file_heuristic = True
            elif has_audio_codec and not is_video_heuristic:
                is_audio_file_heuristic = True
            elif (
                not mime_type.startswith("image/")
                and not is_video_heuristic
                and has_duration
                and (has_sample_rate or has_audio_codec)
            ):
                is_audio_file_heuristic = True

            if is_audio_file_heuristic:
                if not await ecs_service.get_components(session, entity.id, AudioPropertiesComponent):
                    audio_comp = AudioPropertiesComponent()
                    duration = _get_hachoir_metadata(hachoir_metadata, "duration")
                    if duration:
                        audio_comp.duration_seconds = duration.total_seconds()
                    audio_codec_val = _get_hachoir_metadata(hachoir_metadata, "audio_codec")
                    if not audio_codec_val and _has_hachoir_metadata(hachoir_metadata, "compression"):
                        audio_codec_val = _get_hachoir_metadata(hachoir_metadata, "compression")
                    audio_comp.codec_name = audio_codec_val
                    audio_comp.sample_rate_hz = _get_hachoir_metadata(hachoir_metadata, "sample_rate")
                    audio_comp.channels = _get_hachoir_metadata(hachoir_metadata, "nb_channel")
                    bit_rate_bps = _get_hachoir_metadata(hachoir_metadata, "bit_rate")
                    if bit_rate_bps:
                        audio_comp.bit_rate_kbps = bit_rate_bps // 1000
                    await ecs_service.add_component_to_entity(session, entity.id, audio_comp, flush=False)
                    logger.info(f"Added AudioPropertiesComponent for standalone audio Entity ID {entity.id}")

            if is_video_heuristic:
                if not await ecs_service.get_components(session, entity.id, FramePropertiesComponent):
                    video_frame_comp = FramePropertiesComponent()
                    nb_frames = _get_hachoir_metadata(hachoir_metadata, "nb_frames") or _get_hachoir_metadata(
                        hachoir_metadata, "frame_count"
                    )
                    video_frame_comp.frame_count = nb_frames
                    video_frame_comp.nominal_frame_rate = _get_hachoir_metadata(hachoir_metadata, "frame_rate")
                    video_duration = _get_hachoir_metadata(hachoir_metadata, "duration")
                    if video_duration:
                        video_frame_comp.animation_duration_seconds = video_duration.total_seconds()
                    if (
                        not video_frame_comp.frame_count
                        and video_frame_comp.nominal_frame_rate
                        and video_frame_comp.animation_duration_seconds
                        and video_frame_comp.animation_duration_seconds > 0
                    ):
                        video_frame_comp.frame_count = int(
                            video_frame_comp.nominal_frame_rate * video_frame_comp.animation_duration_seconds
                        )
                    await ecs_service.add_component_to_entity(
                        session, entity.id, video_frame_comp, flush=False
                    )
                    logger.info(f"Added FramePropertiesComponent for video/animated Entity ID {entity.id}")

                if has_audio_codec and not await ecs_service.get_components(
                    session, entity.id, AudioPropertiesComponent
                ):
                    video_audio_comp = AudioPropertiesComponent()
                    video_duration_audio = _get_hachoir_metadata(hachoir_metadata, "duration")
                    if video_duration_audio:
                        video_audio_comp.duration_seconds = video_duration_audio.total_seconds()
                    video_audio_comp.codec_name = _get_hachoir_metadata(hachoir_metadata, "audio_codec")
                    video_audio_comp.sample_rate_hz = _get_hachoir_metadata(hachoir_metadata, "sample_rate")
                    video_audio_comp.channels = _get_hachoir_metadata(hachoir_metadata, "nb_channel")
                    await ecs_service.add_component_to_entity(
                        session, entity.id, video_audio_comp, flush=False
                    )
                    logger.info(f"Added AudioPropertiesComponent for video's audio stream, Entity ID {entity.id}")

            if mime_type == "image/gif":
                if not await ecs_service.get_components(session, entity.id, FramePropertiesComponent):
                    frame_comp = FramePropertiesComponent()
                    nb_frames_gif = _get_hachoir_metadata(hachoir_metadata, "nb_frames") or _get_hachoir_metadata(
                        hachoir_metadata, "frame_count"
                    )
                    frame_comp.frame_count = nb_frames_gif
                    duration_gif_obj = _get_hachoir_metadata(hachoir_metadata, "duration")
                    if duration_gif_obj:
                        duration_sec = duration_gif_obj.total_seconds()
                        frame_comp.animation_duration_seconds = duration_sec
                        if nb_frames_gif and nb_frames_gif > 1 and duration_sec > 0:
                            frame_comp.nominal_frame_rate = nb_frames_gif / duration_sec
                    await ecs_service.add_component_to_entity(session, entity.id, frame_comp, flush=False)
                    logger.info(f"Added FramePropertiesComponent for animated GIF Entity ID {entity.id}")


        # Exiftool metadata extraction
        logger.info(f"Attempting Exiftool metadata extraction for {filepath_on_disk} (Entity ID {entity.id})")
        exiftool_data = await _extract_metadata_with_exiftool_async(filepath_on_disk)

        if exiftool_data:
            if not await ecs_service.get_component(session, entity.id, ExiftoolMetadataComponent):
                exif_comp = ExiftoolMetadataComponent(raw_exif_json=exiftool_data)
                await ecs_service.add_component_to_entity(session, entity.id, exif_comp, flush=False)
                logger.info(f"Added ExiftoolMetadataComponent for Entity ID {entity.id}")
            else:
                logger.info(
                    f"ExiftoolMetadataComponent already exists for Entity ID {entity.id}, not adding duplicate."
                )
        else:
            logger.info(f"No metadata extracted by Exiftool for {filepath_on_disk} (Entity ID {entity.id})")

        # Clean up ALL marker components found at the beginning
        # The `current_markers` list was fetched at the start of the loop for this entity.
        for marker_to_remove_loop_var in current_markers: # Use a different loop variable name
            # Double check it still exists before removing.
            # This check might be redundant if remove_component is idempotent or handles missing gracefully.
            # To be safe, fetch by specific marker ID if possible, or re-fetch by entity_id + type if only one should exist.
            # Given the issue, it's safer to iterate what we fetched and try to remove each.
            # The `remove_component` service should ideally take the component instance.

            # Check if the marker (by its specific ID) is still in the session or database before attempting removal
            # This check might be complex if marker_to_remove_loop_var is detached or stale.
            # A simpler approach is to just try removing it, assuming remove_component can handle if it's already gone.
            # Let's assume ecs_service.remove_component can handle being passed a component instance that might be stale
            # or already deleted from the session's perspective, or it re-fetches.
            # To be very safe, we could re-fetch the specific marker by its ID before removing,
            # but that adds DB calls. The current ecs_service.remove_component takes the instance.

            # Let's rely on remove_component to handle the instance correctly.
            # The logger inside remove_component can tell us if it did something.
            await ecs_service.remove_component(session, marker_to_remove_loop_var, flush=False)
            logger.debug(f"Attempted removal of NeedsMetadataExtractionComponent (ID: {marker_to_remove_loop_var.id}) from Entity ID {entity.id}")


    logger.info("MetadataExtractionSystem finished processing entities.")
    # Session flush will be handled by the WorldScheduler after the stage execution.
    # If individual flushes are needed per entity inside the loop (e.g. to release locks sooner),
    # then add session.flush() after each entity's processing, but be mindful of performance.
    # For now, relying on stage-level flush.
