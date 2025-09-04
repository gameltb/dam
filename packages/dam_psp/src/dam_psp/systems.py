import logging

from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam_archive.models import ArchiveMemberComponent
from dam_fs.commands import GetAssetStreamCommand
from dam_fs.events import AssetsReadyForMetadataExtraction
from dam_fs.models import FilePropertiesComponent

from dam_psp.psp_iso_functions import process_iso_stream

from .models import PSPSFOMetadataComponent, PspSfoRawMetadataComponent

logger = logging.getLogger(__name__)


@system(on_event=AssetsReadyForMetadataExtraction)
async def psp_iso_metadata_extraction_system(
    event: AssetsReadyForMetadataExtraction,
    transaction: EcsTransaction,
    world: World,
):
    """
    Listens for assets ready for metadata extraction and processes PSP ISOs.
    """
    logger.info(f"Received {len(event.entity_ids)} assets for PSP ISO metadata extraction.")

    for entity_id in event.entity_ids:
        try:
            # Skip if already processed
            existing_component = await transaction.get_component(entity_id, PSPSFOMetadataComponent)
            if existing_component:
                continue

            # Check if this is an ISO file
            is_iso = False
            archive_member_component = await transaction.get_component(entity_id, ArchiveMemberComponent)
            if archive_member_component:
                if archive_member_component.path_in_archive.lower().endswith(".iso"):
                    is_iso = True
            else:
                file_props = await transaction.get_component(entity_id, FilePropertiesComponent)
                if file_props and file_props.original_filename.lower().endswith(".iso"):
                    is_iso = True

            if not is_iso:
                continue

            # Get the stream
            stream = await world.dispatch_command(GetAssetStreamCommand(entity_id=entity_id))

            if stream:
                with stream:
                    sfo = process_iso_stream(stream)

                if sfo and sfo.data:
                    sfo_metadata = sfo.data
                    sfo_component = PSPSFOMetadataComponent(
                        app_ver=sfo_metadata.get("APP_VER"),
                        bootable=sfo_metadata.get("BOOTABLE"),
                        category=sfo_metadata.get("CATEGORY"),
                        disc_id=sfo_metadata.get("DISC_ID"),
                        disc_version=sfo_metadata.get("DISC_VERSION"),
                        parental_level=sfo_metadata.get("PARENTAL_LEVEL"),
                        psp_system_ver=sfo_metadata.get("PSP_SYSTEM_VER"),
                        title=sfo_metadata.get("TITLE"),
                    )
                    await transaction.add_component_to_entity(entity_id, sfo_component)

                    sfo_raw_component = PspSfoRawMetadataComponent(metadata_json=sfo_metadata)
                    await transaction.add_component_to_entity(entity_id, sfo_raw_component)

                    logger.info(f"Successfully added PSPSFOMetadataComponent to entity {entity_id}.")
                else:
                    logger.warning(f"Could not extract SFO metadata from ISO for entity {entity_id}.")

        except Exception as e:
            logger.error(f"Failed during PSP ISO metadata processing for entity {entity_id}: {e}", exc_info=True)
