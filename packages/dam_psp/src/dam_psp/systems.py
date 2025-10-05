import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncGenerator, AsyncIterator, BinaryIO, Optional, cast

from dam.commands.asset_commands import (
    GetAssetFilenamesCommand,
    GetAssetStreamCommand,
    GetOrCreateEntityFromStreamCommand,
)
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.types import StreamProvider
from dam.core.world import World
from dam.events import AssetReadyForMetadataExtractionEvent
from dam.system_events.entity_events import NewEntityCreatedEvent
from dam.system_events.progress import (
    ProgressCompleted,
    ProgressError,
    ProgressStarted,
    SystemProgressEvent,
)
from sqlalchemy import select

from .commands import (
    CheckCsoIngestionCommand,
    CheckPSPMetadataCommand,
    ClearCsoIngestionCommand,
    ExtractPSPMetadataCommand,
    IngestCsoCommand,
    ReissueVirtualIsoEventCommand,
    RemovePSPMetadataCommand,
)
from .cso_support import CsoDecompressor
from .models import (
    CsoParentIsoComponent,
    IngestedCsoComponent,
    PSPSFOMetadataComponent,
    PspSfoRawMetadataComponent,
)
from .psp_iso_functions import process_iso_stream

logger = logging.getLogger(__name__)


@system(on_command=CheckCsoIngestionCommand)
async def check_cso_ingestion_handler(
    cmd: CheckCsoIngestionCommand,
    transaction: WorldTransaction,
) -> bool:
    """Checks if the IngestedCsoComponent exists for the entity."""
    component = await transaction.get_component(cmd.entity_id, IngestedCsoComponent)
    return component is not None


@system(on_command=ClearCsoIngestionCommand)
async def clear_cso_ingestion_handler(
    cmd: ClearCsoIngestionCommand,
    transaction: WorldTransaction,
):
    """Removes the CSO ingestion components from the entity and its virtual ISO parent."""
    # Remove the marker from the CSO entity
    ingested_comp = await transaction.get_component(cmd.entity_id, IngestedCsoComponent)
    if ingested_comp:
        await transaction.remove_component(ingested_comp)
        logger.info(f"Removed IngestedCsoComponent from CSO entity {cmd.entity_id}")

    # Find and remove the link from the virtual ISO entity
    stmt = select(CsoParentIsoComponent).where(CsoParentIsoComponent.cso_entity_id == cmd.entity_id)
    result = await transaction.session.execute(stmt)
    parent_iso_comp = result.scalar_one_or_none()

    if parent_iso_comp:
        await transaction.remove_component(parent_iso_comp)
        logger.info(
            f"Removed CsoParentIsoComponent from virtual ISO entity {parent_iso_comp.entity_id} "
            f"(linked to CSO {cmd.entity_id})"
        )


@system(on_command=ReissueVirtualIsoEventCommand)
async def reissue_virtual_iso_event_handler(
    cmd: ReissueVirtualIsoEventCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> AsyncGenerator[SystemProgressEvent | NewEntityCreatedEvent, None]:
    """
    Handles re-issuing a NewEntityCreatedEvent for the virtual ISO derived from a CSO file.
    """
    yield ProgressStarted()
    stmt = select(CsoParentIsoComponent).where(CsoParentIsoComponent.cso_entity_id == cmd.entity_id)
    result = await transaction.session.execute(stmt)
    parent_iso_comp = result.scalar_one_or_none()

    if not parent_iso_comp:
        yield ProgressError(
            message=f"No virtual ISO found for CSO entity {cmd.entity_id}",
            exception=ValueError(f"No virtual ISO found for CSO entity {cmd.entity_id}"),
        )
        return

    iso_entity_id = parent_iso_comp.entity_id

    # We need to provide a stream provider for the virtual ISO.
    # This is handled by the get_virtual_iso_asset_stream_handler.
    stream_provider = await get_virtual_iso_asset_stream_handler(
        GetAssetStreamCommand(entity_id=iso_entity_id), transaction, world
    )

    iso_filename = "virtual.iso"
    cso_filenames = await world.dispatch_command(
        GetAssetFilenamesCommand(entity_id=cmd.entity_id)
    ).get_all_results_flat()
    if cso_filenames:
        iso_filename = str(Path(cso_filenames[0]).with_suffix(".iso"))

    yield NewEntityCreatedEvent(
        entity_id=iso_entity_id,
        stream_provider=stream_provider,
        filename=iso_filename,
    )
    yield ProgressCompleted(message=f"Re-issued event for virtual ISO entity {iso_entity_id}")


class DecompressedStreamProvider(StreamProvider):
    def __init__(self, source_provider: StreamProvider):
        self._source_provider = source_provider

    @asynccontextmanager
    async def get_stream(self) -> AsyncIterator[BinaryIO]:
        async with self._source_provider.get_stream() as cso_s:
            yield io.BufferedReader(CsoDecompressor(cso_s))


@system(on_command=IngestCsoCommand)
async def ingest_cso_handler(
    cmd: IngestCsoCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> AsyncGenerator[SystemProgressEvent | NewEntityCreatedEvent, None]:
    """
    Handles ingesting a CSO file by decompressing it into a virtual ISO entity.
    """
    yield ProgressStarted()
    try:
        # 1. Get stream provider for the CSO file
        cso_stream_provider = await cmd.get_stream_provider(world)
        if not cso_stream_provider:
            raise ValueError("Could not get stream provider for CSO file.")

        # 2. Create a provider for the decompressed stream
        decompressed_provider = DecompressedStreamProvider(cso_stream_provider)

        # 3. Get or create the virtual ISO entity from the decompressed stream
        get_or_create_cmd = GetOrCreateEntityFromStreamCommand(stream_provider=decompressed_provider)
        iso_entity, created = await world.dispatch_command(get_or_create_cmd).get_one_value()

        if not iso_entity:
            raise ValueError("Could not get or create entity for virtual ISO.")

        logger.info(f"Virtual ISO entity {'created' if created else 'found'}: {iso_entity.id}")

        # 4. Link the entities together with components
        await transaction.add_or_update_component(iso_entity.id, CsoParentIsoComponent(cso_entity_id=cmd.entity_id))
        await transaction.add_or_update_component(cmd.entity_id, IngestedCsoComponent())

        # 5. Emit a NewEntityCreatedEvent for the virtual ISO
        iso_filename = "virtual.iso"
        cso_filenames = await world.dispatch_command(
            GetAssetFilenamesCommand(entity_id=cmd.entity_id)
        ).get_all_results_flat()
        if cso_filenames:
            iso_filename = str(Path(cso_filenames[0]).with_suffix(".iso"))

        yield NewEntityCreatedEvent(
            entity_id=iso_entity.id,
            stream_provider=decompressed_provider,  # Reuse the provider
            filename=iso_filename,
        )

        yield ProgressCompleted(message="CSO ingestion complete.")

    except Exception as e:
        logger.error(f"Failed during CSO ingestion for entity {cmd.entity_id}: {e}", exc_info=True)
        yield ProgressError(message=str(e), exception=e)


@system(on_command=CheckPSPMetadataCommand)
async def check_psp_metadata_handler(
    cmd: CheckPSPMetadataCommand,
    transaction: WorldTransaction,
) -> bool:
    """Checks if the PSPSFOMetadataComponent exists for the entity."""
    component = await transaction.get_component(cmd.entity_id, PSPSFOMetadataComponent)
    return component is not None


@system(on_command=RemovePSPMetadataCommand)
async def remove_psp_metadata_handler(
    cmd: RemovePSPMetadataCommand,
    transaction: WorldTransaction,
):
    """Removes the PSP metadata components from the entity."""
    sfo_comp = await transaction.get_component(cmd.entity_id, PSPSFOMetadataComponent)
    if sfo_comp:
        await transaction.remove_component(sfo_comp)

    sfo_raw_comp = await transaction.get_component(cmd.entity_id, PspSfoRawMetadataComponent)
    if sfo_raw_comp:
        await transaction.remove_component(sfo_raw_comp)

    logger.info(f"Removed PSP metadata components from entity {cmd.entity_id}")


@system(on_event=AssetReadyForMetadataExtractionEvent)
async def psp_iso_metadata_extraction_event_handler_system(
    event: AssetReadyForMetadataExtractionEvent,
    transaction: WorldTransaction,
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
            all_filenames = await world.dispatch_command(
                GetAssetFilenamesCommand(entity_id=entity_id)
            ).get_all_results_flat()

            is_iso = any(filename.lower().endswith(".iso") for filename in all_filenames)

            if is_iso:
                await world.dispatch_command(ExtractPSPMetadataCommand(entity_id=entity_id)).get_all_results()

        except Exception as e:
            logger.error(f"Failed during PSP ISO metadata processing for entity {entity_id}: {e}", exc_info=True)


@system(on_command=ExtractPSPMetadataCommand)
async def psp_iso_metadata_extraction_command_handler_system(
    command: ExtractPSPMetadataCommand,
    transaction: WorldTransaction,
    world: World,
) -> None:
    """
    Handles the command to extract metadata from a PSP ISO file.
    """
    entity_id = command.entity_id
    try:
        provider = await command.get_stream_provider(world)
        if not provider:
            logger.warning(f"Could not get asset stream for PSP ISO for entity {entity_id}.")
            return

        async with provider.get_stream() as stream:
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
                await transaction.add_or_update_component(entity_id, sfo_component)

                sfo_raw_component = PspSfoRawMetadataComponent(metadata_json=sfo_metadata)
                await transaction.add_or_update_component(entity_id, sfo_raw_component)

                logger.info(f"Successfully added or updated PSPSFOMetadataComponent for entity {entity_id}.")
            else:
                logger.warning(f"Could not extract SFO metadata from ISO for entity {entity_id}.")
    except Exception as e:
        logger.error(f"Failed during PSP ISO metadata processing for entity {entity_id}: {e}", exc_info=True)


@system(on_command=GetAssetStreamCommand)
async def get_virtual_iso_asset_stream_handler(
    cmd: GetAssetStreamCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
) -> Optional[StreamProvider]:
    """
    Handles getting a stream provider for a virtual ISO entity derived from a CSO.
    """
    # 1. Check if the entity is a virtual ISO
    parent_iso_comp = await transaction.get_component(cmd.entity_id, CsoParentIsoComponent)
    if not parent_iso_comp:
        return None  # This is not a virtual ISO, so another handler will take care of it.

    logger.info(
        f"Entity {cmd.entity_id} is a virtual ISO. Getting stream from CSO entity {parent_iso_comp.cso_entity_id}."
    )

    # 2. Get the stream provider for the parent CSO file.
    cso_stream_cmd = GetAssetStreamCommand(entity_id=parent_iso_comp.cso_entity_id)
    try:
        cso_stream_provider = await world.dispatch_command(cso_stream_cmd).get_first_non_none_value()
    except ValueError:
        cso_stream_provider = None

    if not cso_stream_provider:
        logger.error(f"Could not get stream provider for parent CSO {parent_iso_comp.cso_entity_id}")
        return None

    # 3. Create a new stream provider that returns the decompressed stream.
    class DecompressedStreamProvider(StreamProvider):
        def __init__(self, source_provider: StreamProvider):
            self._source_provider = source_provider

        @asynccontextmanager
        async def get_stream(self) -> AsyncIterator[BinaryIO]:
            async with self._source_provider.get_stream() as cso_s:
                yield io.BufferedReader(CsoDecompressor(cso_s))

    return DecompressedStreamProvider(cso_stream_provider)
