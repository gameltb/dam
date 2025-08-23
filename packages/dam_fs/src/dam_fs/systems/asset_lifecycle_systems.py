import binascii  # For hex string to bytes conversion
import logging
from typing import Optional

from dam.core.config import WorldConfig
from ..events import (
    AssetFileIngestionRequested,
    AssetReferenceIngestionRequested,
    FindEntityByHashQuery,
)
from dam.core.system_params import WorldSession
from dam.core.systems import listens_for
from dam.core.world import get_world
from dam.models.core.entity import Entity
from dam_fs.models.file_location_component import FileLocationComponent
from dam.models.hashes.content_hash_md5_component import ContentHashMD5Component
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam_fs.models.file_properties_component import FilePropertiesComponent
from dam.services import ecs_service
from dam_source.services import import_service

logger = logging.getLogger(__name__)

# --- Command Systems (Event Handlers for Ingestion) ---


@listens_for(AssetFileIngestionRequested)
async def handle_asset_file_ingestion_request(event: AssetFileIngestionRequested):
    """
    Handles the ingestion of an asset file by copying it, forwarding to the import_service.
    """
    logger.info(
        f"System handling AssetFileIngestionRequested for: {event.original_filename} in world {event.world_name}"
    )
    try:
        world = get_world(event.world_name)
        if not world:
            raise import_service.ImportServiceError(f"World '{event.world_name}' not found.")

        await import_service.import_local_file(
            world=world,
            filepath=event.filepath_on_disk,
            copy_to_storage=True,
            original_filename=event.original_filename,
            size_bytes=event.size_bytes,
        )
        logger.info(f"Successfully processed AssetFileIngestionRequested for {event.original_filename}")
    except import_service.ImportServiceError as e:
        logger.error(f"Failed to process AssetFileIngestionRequested for {event.original_filename}: {e}", exc_info=True)


@listens_for(AssetReferenceIngestionRequested)
async def handle_asset_reference_ingestion_request(event: AssetReferenceIngestionRequested):
    """
    Handles the ingestion of an asset by reference, forwarding to the import_service.
    """
    logger.info(
        f"System handling AssetReferenceIngestionRequested for: {event.original_filename} in world {event.world_name}"
    )
    try:
        world = get_world(event.world_name)
        if not world:
            raise import_service.ImportServiceError(f"World '{event.world_name}' not found.")

        await import_service.import_local_file(
            world=world,
            filepath=event.filepath_on_disk,
            copy_to_storage=False,
            original_filename=event.original_filename,
            size_bytes=event.size_bytes,
        )
        logger.info(f"Successfully processed AssetReferenceIngestionRequested for {event.original_filename}")
    except import_service.ImportServiceError as e:
        logger.error(
            f"Failed to process AssetReferenceIngestionRequested for {event.original_filename}: {e}", exc_info=True
        )


@listens_for(FindEntityByHashQuery)
async def handle_find_entity_by_hash_query(
    event: FindEntityByHashQuery,
    session: WorldSession,
    world_config: WorldConfig,
):
    logger.info(
        f"System handling FindEntityByHashQuery for hash: {event.hash_value} (type: {event.hash_type}) in world '{world_config.name}' (Req ID: {event.request_id})"
    )
    if not event.result_future:
        logger.error(
            f"Result future not set on FindEntityByHashQuery event (Req ID: {event.request_id}). Cannot proceed."
        )
        return

    try:
        try:
            hash_bytes = binascii.unhexlify(event.hash_value)
        except binascii.Error as e:
            logger.error(
                f"[QueryResult RequestID: {event.request_id}] Invalid hex string for hash_value '{event.hash_value}': {e}"
            )
            raise ValueError(
                f"Invalid hash_value format: {event.hash_value}"
            ) from e

        entity = await ecs_service.find_entity_by_content_hash(session, hash_bytes, event.hash_type)
        entity_details_dict = None

        if entity:
            logger.info(
                f"[QueryResult RequestID: {event.request_id}] Found Entity ID: {entity.id} for hash {event.hash_value}"
            )
            entity_details_dict = {"entity_id": entity.id, "components": {}}

            fpc = await ecs_service.get_component(session, entity.id, FilePropertiesComponent)
            if fpc:
                entity_details_dict["components"]["FilePropertiesComponent"] = [
                    {
                        "original_filename": fpc.original_filename,
                        "file_size_bytes": fpc.file_size_bytes,
                    }
                ]

            flcs = await ecs_service.get_components(session, entity.id, FileLocationComponent)
            if flcs:
                entity_details_dict["components"]["FileLocationComponent"] = [
                    {
                        "content_identifier": flc.content_identifier,
                        "url": flc.url,
                    }
                    for flc in flcs
                ]

            sha256_comp = await ecs_service.get_component(session, entity.id, ContentHashSHA256Component)
            if sha256_comp:
                entity_details_dict["components"]["ContentHashSHA256Component"] = [
                    {"hash_value": sha256_comp.hash_value.hex()}
                ]

            md5_comp = await ecs_service.get_component(session, entity.id, ContentHashMD5Component)
            if md5_comp:
                entity_details_dict["components"]["ContentHashMD5Component"] = [
                    {"hash_value": md5_comp.hash_value.hex()}
                ]
        else:
            logger.info(f"[QueryResult RequestID: {event.request_id}] No entity found for hash {event.hash_value}")

        if not event.result_future.done():
            event.result_future.set_result(entity_details_dict)

    except Exception as e:
        logger.error(f"Error in handle_find_entity_by_hash_query (Req ID: {event.request_id}): {e}", exc_info=True)
        if not event.result_future.done():
            event.result_future.set_exception(e)

__all__ = [
    "handle_asset_file_ingestion_request",
    "handle_asset_reference_ingestion_request",
    "handle_find_entity_by_hash_query",
]
