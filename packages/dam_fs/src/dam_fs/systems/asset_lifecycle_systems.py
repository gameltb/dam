import binascii  # For hex string to bytes conversion
import logging
from typing import Optional

from dam.core.config import WorldConfig
from dam.core.systems import handles_command
from dam.core.transaction import EcsTransaction
from dam.core.world import get_world
from dam.models.core.entity import Entity
from dam.models.hashes.content_hash_md5_component import ContentHashMD5Component
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam.functions import ecs_functions
from dam_source.functions import import_functions

from ..commands import (
    FindEntityByHashCommand,
    IngestFileCommand,
    IngestReferenceCommand,
)
from ..models.file_location_component import FileLocationComponent
from ..models.file_properties_component import FilePropertiesComponent

logger = logging.getLogger(__name__)


@handles_command(IngestFileCommand)
async def handle_ingest_file_command(cmd: IngestFileCommand, transaction: EcsTransaction):
    """
    Handles the command to ingest an asset file by copying it.
    """
    logger.info(f"System handling IngestFileCommand for: {cmd.original_filename} in world {cmd.world_name}")
    try:
        # The function still needs the world for resources, which is a design smell to be fixed.
        world = get_world(cmd.world_name)
        if not world:
            raise import_functions.ImportServiceError(f"World '{cmd.world_name}' not found.")

        entity = await import_functions.import_local_file(
            world=world,
            transaction=transaction,
            filepath=cmd.filepath_on_disk,
            copy_to_storage=True,
            original_filename=cmd.original_filename,
            size_bytes=cmd.size_bytes,
        )
        logger.info(f"Successfully processed IngestFileCommand for {cmd.original_filename}")

        from dam_media_audio.commands import ExtractAudioMetadataCommand
        command = ExtractAudioMetadataCommand(entity=entity)
        logger.info("Dispatching ExtractAudioMetadataCommand")
        await world.dispatch_command(command)
        logger.info("Waiting for ExtractAudioMetadataCommand future")
        await command.result_future
        logger.info("ExtractAudioMetadataCommand future finished")
    except import_functions.ImportServiceError as e:
        logger.error(f"Failed to process IngestFileCommand for {cmd.original_filename}: {e}", exc_info=True)


@handles_command(IngestReferenceCommand)
async def handle_ingest_reference_command(cmd: IngestReferenceCommand, transaction: EcsTransaction):
    """
    Handles the command to ingest an asset by reference.
    """
    logger.info(f"System handling IngestReferenceCommand for: {cmd.original_filename} in world {cmd.world_name}")
    try:
        # The function still needs the world for resources, which is a design smell to be fixed.
        world = get_world(cmd.world_name)
        if not world:
            raise import_functions.ImportServiceError(f"World '{cmd.world_name}' not found.")

        await import_functions.import_local_file(
            world=world,
            transaction=transaction,
            filepath=cmd.filepath_on_disk,
            copy_to_storage=False,
            original_filename=cmd.original_filename,
            size_bytes=cmd.size_bytes,
        )
        logger.info(f"Successfully processed IngestReferenceCommand for {cmd.original_filename}")
    except import_functions.ImportServiceError as e:
        logger.error(f"Failed to process IngestReferenceCommand for {cmd.original_filename}: {e}", exc_info=True)


@handles_command(FindEntityByHashCommand)
async def handle_find_entity_by_hash_command(
    cmd: FindEntityByHashCommand,
    transaction: EcsTransaction,
    world_config: WorldConfig,
):
    """
    Handles the command to find an entity by its content hash.
    """
    logger.info(
        f"System handling FindEntityByHashCommand for hash: {cmd.hash_value} (type: {cmd.hash_type}) in world '{world_config.name}' (Req ID: {cmd.request_id})"
    )
    if not cmd.result_future:
        logger.error(f"Result future not set on FindEntityByHashCommand (Req ID: {cmd.request_id}). Cannot proceed.")
        return

    try:
        try:
            hash_bytes = binascii.unhexlify(cmd.hash_value)
        except binascii.Error as e:
            logger.error(
                f"[QueryResult RequestID: {cmd.request_id}] Invalid hex string for hash_value '{cmd.hash_value}': {e}"
            )
            raise ValueError(f"Invalid hash_value format: {cmd.hash_value}") from e

        # I need to add find_entity_by_content_hash to the EcsTransaction wrapper
        entity = await ecs_functions.find_entity_by_content_hash(transaction.session, hash_bytes, cmd.hash_type)
        entity_details_dict = None

        if entity:
            logger.info(f"[QueryResult RequestID: {cmd.request_id}] Found Entity ID: {entity.id} for hash {cmd.hash_value}")
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
                entity_details_dict["components"]["ContentHashMD5Component"] = [{"hash_value": md5_comp.hash_value.hex()}]
        else:
            logger.info(f"[QueryResult RequestID: {cmd.request_id}] No entity found for hash {cmd.hash_value}")

        if not cmd.result_future.done():
            cmd.result_future.set_result(entity_details_dict)

    except Exception as e:
        logger.error(f"Error in handle_find_entity_by_hash_command (Req ID: {cmd.request_id}): {e}", exc_info=True)
        if not cmd.result_future.done():
            cmd.result_future.set_exception(e)


__all__ = [
    "handle_ingest_file_command",
    "handle_ingest_reference_command",
    "handle_find_entity_by_hash_command",
]
