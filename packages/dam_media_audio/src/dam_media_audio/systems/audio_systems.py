"""Defines systems for handling audio media."""

import asyncio
import json
import logging
import subprocess
from typing import Any

from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam_fs.functions import file_operations
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.utils.url_utils import get_local_path_for_url

from ..commands import ExtractAudioMetadataCommand
from ..models.properties.audio_properties_component import AudioPropertiesComponent

logger = logging.getLogger(__name__)


def _parse_ffprobe_output_and_create_component(metadata: dict[str, Any]) -> AudioPropertiesComponent | None:
    """
    Parse ffprobe output and create an AudioPropertiesComponent.

    Args:
        metadata: The JSON output from ffprobe.

    Returns:
        An AudioPropertiesComponent if an audio stream is found, otherwise None.

    """
    audio_stream = next(
        (stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "audio"),
        None,
    )

    if not audio_stream:
        return None

    audio_comp = AudioPropertiesComponent()

    duration = audio_stream.get("duration", metadata.get("format", {}).get("duration"))
    if duration:
        audio_comp.duration_seconds = float(duration)

    audio_comp.codec_name = audio_stream.get("codec_name")

    sample_rate = audio_stream.get("sample_rate")
    if sample_rate:
        audio_comp.sample_rate_hz = int(sample_rate)

    channels = audio_stream.get("channels")
    if channels:
        audio_comp.channels = int(channels)

    bit_rate = audio_stream.get("bit_rate", metadata.get("format", {}).get("bit_rate"))
    if bit_rate:
        audio_comp.bit_rate_kbps = int(bit_rate) // 1000

    return audio_comp


@system(on_command=ExtractAudioMetadataCommand)
async def add_audio_components_system(
    cmd: ExtractAudioMetadataCommand,
    transaction: WorldTransaction,
) -> bool:
    """
    Extract metadata from an audio file and add it as components to the entity.

    This system uses ffprobe to get audio properties and adds an
    AudioPropertiesComponent to the entity.
    """
    logger.info("Running add_audio_components_system with ffprobe")

    entity = cmd.entity
    all_locations = await transaction.get_components(entity.id, FileLocationComponent)
    if not all_locations:
        return False

    filepath_on_disk = None
    for loc in all_locations:
        try:
            potential_path = get_local_path_for_url(loc.url)
            is_file = await asyncio.to_thread(potential_path.is_file) if potential_path else False
            if potential_path and is_file:
                filepath_on_disk = potential_path
                break
        except (ValueError, FileNotFoundError):
            continue

    if not filepath_on_disk:
        return False

    mime_type = await file_operations.get_mime_type_async(filepath_on_disk)
    logger.info("MIME type for %s: %s", filepath_on_disk, mime_type)
    if not mime_type.startswith("audio/"):
        return False

    try:
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(filepath_on_disk),
        ]
        result = await asyncio.to_thread(subprocess.run, ffprobe_cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning("ffprobe failed for %s: %s", filepath_on_disk, e)
        return False

    if await transaction.get_components(entity.id, AudioPropertiesComponent):
        return False

    audio_comp = _parse_ffprobe_output_and_create_component(metadata)

    if not audio_comp:
        logger.warning("No audio stream found in %s", filepath_on_disk)
        return False

    await transaction.add_component_to_entity(entity.id, audio_comp)
    logger.info("Added AudioPropertiesComponent for standalone audio Entity ID %s", entity.id)

    await transaction.flush()

    return True
