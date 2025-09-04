import binascii  # For hex string to bytes conversion
import io
import logging
from typing import Annotated

from dam.commands import GetAssetFilenamesCommand
from dam.core.commands import GetOrCreateEntityFromStreamCommand
from dam.core.config import WorldConfig
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.functions import ecs_functions
from dam.models.hashes.content_hash_md5_component import ContentHashMD5Component
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam_source.models.source_info import source_types
from dam_source.models.source_info.original_source_info_component import OriginalSourceInfoComponent

from ..commands import (
    AddFilePropertiesCommand,
    FindEntityByHashCommand,
    IngestFileCommand,
)
from ..models.file_location_component import FileLocationComponent
from ..models.file_properties_component import FilePropertiesComponent
from ..resources.file_storage_resource import FileStorageResource

logger = logging.getLogger(__name__)


@system(on_command=AddFilePropertiesCommand)
async def add_file_properties_handler(
    cmd: AddFilePropertiesCommand,
    transaction: EcsTransaction,
):
    """
    Handles adding file properties to an entity.
    """
    logger.info(f"System handling AddFilePropertiesCommand for entity: {cmd.entity_id}")
    fpc = FilePropertiesComponent(original_filename=cmd.original_filename, file_size_bytes=cmd.size_bytes)
    await transaction.add_component_to_entity(cmd.entity_id, fpc)


@system(on_command=IngestFileCommand)
async def handle_ingest_file_command(
    cmd: IngestFileCommand, transaction: EcsTransaction, world: Annotated[World, "Resource"]
):
    """
    Handles the command to ingest an asset file by copying it.
    """
    logger.info(f"System handling IngestFileCommand for: {cmd.original_filename} in world {world.name}")
    try:
        with open(cmd.filepath_on_disk, "rb") as f:
            file_content_stream = io.BytesIO(f.read())

        # 1. Get or create entity from stream
        get_or_create_cmd = GetOrCreateEntityFromStreamCommand(
            stream=file_content_stream,
        )
        command_result = await world.dispatch_command(get_or_create_cmd)
        entity, sha256_bytes = command_result.get_one_value()

        # 2. Add file properties
        add_props_cmd = AddFilePropertiesCommand(
            entity_id=entity.id,
            original_filename=cmd.original_filename,
            size_bytes=cmd.size_bytes,
        )
        await world.dispatch_command(add_props_cmd)

        # 3. Store the file and add components
        file_storage = world.get_resource(FileStorageResource)
        file_content_stream.seek(0)
        _, relative_path = file_storage.store_file(file_content_stream.read(), original_filename=cmd.original_filename)

        absolute_path = file_storage.get_world_asset_storage_path() / relative_path
        url = absolute_path.as_uri()
        source_type = source_types.SOURCE_TYPE_LOCAL_FILE

        existing_flcs = await transaction.get_components(entity.id, FileLocationComponent)
        if not any(flc.url == url for flc in existing_flcs):
            flc = FileLocationComponent(url=url)
            await transaction.add_component_to_entity(entity.id, flc)

        existing_osis = await transaction.get_components_by_value(
            entity.id, OriginalSourceInfoComponent, {"source_type": source_type}
        )
        if not existing_osis:
            osi = OriginalSourceInfoComponent(source_type=source_type)
            await transaction.add_component_to_entity(entity.id, osi)

        logger.info(f"Successfully processed IngestFileCommand for {cmd.original_filename}")
    except Exception as e:
        logger.error(f"Failed to process IngestFileCommand for {cmd.original_filename}: {e}", exc_info=True)


@system(on_command=FindEntityByHashCommand)
async def handle_find_entity_by_hash_command(
    cmd: FindEntityByHashCommand,
    transaction: EcsTransaction,
    world_config: WorldConfig,
) -> dict | None:
    """
    Handles the command to find an entity by its content hash.
    """
    logger.info(
        f"System handling FindEntityByHashCommand for hash: {cmd.hash_value} (type: {cmd.hash_type}) in world '{world_config.name}' (Req ID: {cmd.request_id})"
    )

    try:
        try:
            hash_bytes = binascii.unhexlify(cmd.hash_value)
        except binascii.Error as e:
            logger.error(
                f"[QueryResult RequestID: {cmd.request_id}] Invalid hex string for hash_value '{cmd.hash_value}': {e}"
            )
            raise ValueError(f"Invalid hash_value format: {cmd.hash_value}") from e

        entity = await ecs_functions.find_entity_by_content_hash(transaction.session, hash_bytes, cmd.hash_type)
        entity_details_dict = None

        if entity:
            logger.info(
                f"[QueryResult RequestID: {cmd.request_id}] Found Entity ID: {entity.id} for hash {cmd.hash_value}"
            )
            entity_details_dict = {"entity_id": entity.id, "components": {}}

            fpc = await transaction.get_component(entity.id, FilePropertiesComponent)
            if fpc:
                entity_details_dict["components"]["FilePropertiesComponent"] = [
                    {
                        "original_filename": fpc.original_filename,
                        "file_size_bytes": fpc.file_size_bytes,
                    }
                ]

            flcs = await transaction.get_components(entity.id, FileLocationComponent)
            if flcs:
                entity_details_dict["components"]["FileLocationComponent"] = [
                    {
                        "content_identifier": flc.content_identifier,
                        "url": flc.url,
                    }
                    for flc in flcs
                ]

            sha256_comp = await transaction.get_component(entity.id, ContentHashSHA256Component)
            if sha256_comp:
                entity_details_dict["components"]["ContentHashSHA256Component"] = [
                    {"hash_value": sha256_comp.hash_value.hex()}
                ]

            md5_comp = await transaction.get_component(entity.id, ContentHashMD5Component)
            if md5_comp:
                entity_details_dict["components"]["ContentHashMD5Component"] = [
                    {"hash_value": md5_comp.hash_value.hex()}
                ]
        else:
            logger.info(f"[QueryResult RequestID: {cmd.request_id}] No entity found for hash {cmd.hash_value}")

        return entity_details_dict

    except Exception as e:
        logger.error(f"Error in handle_find_entity_by_hash_command (Req ID: {cmd.request_id}): {e}", exc_info=True)
        raise


@system(on_command=GetAssetFilenamesCommand)
async def get_fs_asset_filenames_handler(
    cmd: GetAssetFilenamesCommand,
    transaction: EcsTransaction,
):
    """
    Handles getting filenames for assets with FilePropertiesComponent.
    """
    file_props = await transaction.get_component(cmd.entity_id, FilePropertiesComponent)
    if file_props and file_props.original_filename:
        # This system returns a list with one filename
        return [file_props.original_filename]
    # This system does not handle this entity if it has no FilePropertiesComponent
    return None
