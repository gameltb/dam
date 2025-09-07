import asyncio
import json
import logging
import subprocess

from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam_fs.functions import file_operations
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.utils.url_utils import get_local_path_for_url

from dam_media_audio.models.properties.audio_properties_component import AudioPropertiesComponent

from ..commands import ExtractAudioMetadataCommand

logger = logging.getLogger(__name__)


@system(on_command=ExtractAudioMetadataCommand)
async def add_audio_components_system(
    cmd: ExtractAudioMetadataCommand,
    transaction: EcsTransaction,
) -> None:
    logger.info("Running add_audio_components_system with ffprobe")

    entity = cmd.entity
    all_locations = await transaction.get_components(entity.id, FileLocationComponent)
    if not all_locations:
        return

    filepath_on_disk = None
    for loc in all_locations:
        try:
            potential_path = get_local_path_for_url(loc.url)
            is_file = False
            if potential_path:
                is_file = await asyncio.to_thread(potential_path.is_file)
            if potential_path and is_file:
                filepath_on_disk = potential_path
                break
        except (ValueError, FileNotFoundError):
            continue

    if not filepath_on_disk:
        return

    mime_type = await file_operations.get_mime_type_async(filepath_on_disk)
    logger.info(f"MIME type for {filepath_on_disk}: {mime_type}")
    if not mime_type.startswith("audio/"):
        return

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
        logger.warning(f"ffprobe failed for {filepath_on_disk}: {e}")
        return

    audio_stream = next(
        (stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "audio"),
        None,
    )

    if not audio_stream:
        logger.warning(f"No audio stream found in {filepath_on_disk}")
        return

    if not await transaction.get_components(entity.id, AudioPropertiesComponent):
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

        await transaction.add_component_to_entity(entity.id, audio_comp)
        logger.info(f"Added AudioPropertiesComponent for standalone audio Entity ID {entity.id}")

    await transaction.flush()
