import asyncio
import logging
from typing import Annotated, List

from dam.core.components_markers import NeedsMetadataExtractionComponent
from dam.core.config import WorldConfig
from dam.core.stages import SystemStage
from dam.core.system_params import WorldSession
from dam.core.systems import system
from dam.models.core.entity import Entity
from dam_fs.models.file_location_component import FileLocationComponent
from dam.services import ecs_service, file_operations
from dam.utils.url_utils import get_local_path_for_url

from dam_media_audio.models.properties.audio_properties_component import AudioPropertiesComponent

try:
    from hachoir.core import config as HachoirConfig
    from hachoir.metadata import extractMetadata
    from hachoir.parser import createParser

    HachoirConfig.quiet = True
    _hachoir_available = True
except ImportError:
    _hachoir_available = False


logger = logging.getLogger(__name__)

def _has_hachoir_metadata(md, key: str) -> bool:
    """Safely checks if Hachoir metadata object has a given key."""
    try:
        return md.has(key)
    except (KeyError, ValueError):
        return False


def _get_hachoir_metadata(md, key: str, default=None) -> any:
    """Safely gets a value from Hachoir metadata object for a given key."""
    try:
        if md.has(key):
            return md.get(key)
    except (KeyError, ValueError):
        pass
    return default

@system(stage=SystemStage.METADATA_EXTRACTION)
async def add_audio_components_system(
    session: WorldSession,
    world_config: WorldConfig,
    entities_to_process: Annotated[List[Entity], "MarkedEntityList", NeedsMetadataExtractionComponent],
):
    if not _hachoir_available:
        logger.warning("Hachoir library not installed. Skipping audio metadata extraction system.")
        return

    if not entities_to_process:
        return

    for entity in entities_to_process:
        all_locations = await ecs_service.get_components(session, entity.id, FileLocationComponent)
        if not all_locations:
            continue

        filepath_on_disk = None
        for loc in all_locations:
            try:
                potential_path = get_local_path_for_url(loc.url, world_config)
                if potential_path and await asyncio.to_thread(potential_path.is_file):
                    filepath_on_disk = potential_path
                    break
            except (ValueError, FileNotFoundError):
                continue

        if not filepath_on_disk:
            continue

        mime_type = await file_operations.get_mime_type_async(filepath_on_disk)
        if not mime_type.startswith("audio/"):
            continue

        parser = createParser(str(filepath_on_disk))
        if not parser:
            continue

        with parser:
            try:
                hachoir_metadata = extractMetadata(parser)
            except Exception:
                hachoir_metadata = None

        if not hachoir_metadata:
            continue

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

    await session.flush()
