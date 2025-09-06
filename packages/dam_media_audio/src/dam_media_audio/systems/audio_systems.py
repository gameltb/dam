import asyncio
import logging
from datetime import timedelta
from typing import Any

from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam_fs.functions import file_operations
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.utils.url_utils import get_local_path_for_url

from dam_media_audio.models.properties.audio_properties_component import AudioPropertiesComponent

from ..commands import ExtractAudioMetadataCommand

try:
    from hachoir.core import config as HachoirConfig  # type: ignore[import]
    from hachoir.metadata import Metadata, extractMetadata  # type: ignore[import]
    from hachoir.parser import createParser  # type: ignore[import]

    HachoirConfig.quiet = True
    _hachoir_available = True
except ImportError:
    _hachoir_available = False


logger = logging.getLogger(__name__)


def _has_hachoir_metadata(md: Metadata, key: str) -> bool:
    """Safely checks if Hachoir metadata object has a given key."""
    try:
        return md.has(key)
    except (KeyError, ValueError):
        return False


def _get_hachoir_metadata(md: Metadata, key: str, default: Any = None) -> Any:
    """Safely gets a value from Hachoir metadata object for a given key."""
    try:
        if md.has(key):
            return md.get(key)
    except (KeyError, ValueError):
        pass
    return default


@system(on_command=ExtractAudioMetadataCommand)
async def add_audio_components_system(
    cmd: ExtractAudioMetadataCommand,
    transaction: EcsTransaction,
) -> None:
    logger.info("Running add_audio_components_system")
    if not _hachoir_available:
        logger.warning("Hachoir library not installed. Skipping audio metadata extraction system.")
        return

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

    parser = createParser(str(filepath_on_disk))
    if not parser:
        return

    with parser:
        try:
            hachoir_metadata = extractMetadata(parser)
        except Exception:
            hachoir_metadata = None

    if not hachoir_metadata:
        return

    if not await transaction.get_components(entity.id, AudioPropertiesComponent):
        audio_comp = AudioPropertiesComponent()
        duration = _get_hachoir_metadata(hachoir_metadata, "duration")
        if isinstance(duration, timedelta):
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
        await transaction.add_component_to_entity(entity.id, audio_comp)
        logger.info(f"Added AudioPropertiesComponent for standalone audio Entity ID {entity.id}")

    await transaction.flush()
