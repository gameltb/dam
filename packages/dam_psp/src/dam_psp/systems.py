import logging
from typing import Optional, cast

from dam.commands import GetAssetFilenamesCommand, GetAssetStreamCommand
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.events import AssetReadyForMetadataExtractionEvent

from dam_psp.commands import ExtractPSPMetadataCommand
from dam_psp.psp_iso_functions import process_iso_stream

from .models import PSPSFOMetadataComponent, PspSfoRawMetadataComponent

logger = logging.getLogger(__name__)


@system(on_event=AssetReadyForMetadataExtractionEvent)
async def psp_iso_metadata_extraction_event_handler_system(
    event: AssetReadyForMetadataExtractionEvent,
    transaction: EcsTransaction,
    world: World,
) -> None:
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

            # Get all possible filenames for the asset
            results = await world.dispatch_command(GetAssetFilenamesCommand(entity_id=entity_id)).get_all_results()
            all_filenames = [item for sublist in results for item in sublist]

            is_iso = any(filename.lower().endswith(".iso") for filename in all_filenames)

            if is_iso:
                await world.dispatch_command(ExtractPSPMetadataCommand(entity_id=entity_id, depth=0)).get_all_results()

        except Exception as e:
            logger.error(f"Failed during PSP ISO metadata processing for entity {entity_id}: {e}", exc_info=True)


@system(on_command=ExtractPSPMetadataCommand)
async def psp_iso_metadata_extraction_command_handler_system(
    command: ExtractPSPMetadataCommand,
    transaction: EcsTransaction,
    world: World,
) -> None:
    """
    Handles the command to extract metadata from a PSP ISO file.
    """
    entity_id = command.entity_id
    try:
        # Get the stream
        stream = await world.dispatch_command(GetAssetStreamCommand(entity_id=entity_id)).get_first_non_none_value()

        if stream:
            with stream:
                sfo = process_iso_stream(stream)

                if sfo and sfo.data:
                    sfo_metadata = sfo.data
                    sfo_component = PSPSFOMetadataComponent(
                        app_ver=cast(Optional[str], sfo_metadata.get("APP_VER")),
                        bootable=cast(Optional[int], sfo_metadata.get("BOOTABLE")),
                        category=cast(Optional[str], sfo_metadata.get("CATEGORY")),
                        disc_id=cast(Optional[str], sfo_metadata.get("DISC_ID")),
                        disc_version=cast(Optional[str], sfo_metadata.get("DISC_VERSION")),
                        parental_level=cast(Optional[int], sfo_metadata.get("PARENTAL_LEVEL")),
                        psp_system_ver=cast(Optional[str], sfo_metadata.get("PSP_SYSTEM_VER")),
                        title=cast(Optional[str], sfo_metadata.get("TITLE")),
                    )
                    await transaction.add_component_to_entity(entity_id, sfo_component)

                    sfo_raw_component = PspSfoRawMetadataComponent(metadata_json=sfo_metadata)
                    await transaction.add_component_to_entity(entity_id, sfo_raw_component)

                    logger.info(f"Successfully added PSPSFOMetadataComponent to entity {entity_id}.")
                else:
                    logger.warning(f"Could not extract SFO metadata from ISO for entity {entity_id}.")
        else:
            logger.warning(f"Could not get asset stream for PSP ISO for entity {entity_id}.")

    except Exception as e:
        logger.error(f"Failed during PSP ISO metadata processing for entity {entity_id}: {e}", exc_info=True)
