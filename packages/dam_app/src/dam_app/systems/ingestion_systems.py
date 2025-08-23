import logging
from typing import Annotated

from dam.core.system_params import WorldSession
from dam.core.systems import listens_for
from dam.core.world import World
from dam_fs.events import FileStored
from dam_fs.resources import FileStorageResource

from ..events import AssetStreamIngestionRequested

logger = logging.getLogger(__name__)


@listens_for(AssetStreamIngestionRequested)
async def ingestion_request_system(
    event: AssetStreamIngestionRequested,
    world: Annotated[World, "Resource"],
    session: WorldSession,
    fs_resource: Annotated[FileStorageResource, "Resource"],
):
    """
    Handles the initial ingestion request for an in-memory asset stream.
    This system is responsible for:
    1. Calculating content hashes.
    2. Storing the file to the Content-Addressable Storage (CAS).
    3. Creating the core `File` component.
    4. Firing a `FileStored` event to trigger the next stage of the pipeline.
    """
    from dam.services import ecs_service, hashing_service

    logger.info(f"Received asset ingestion request for '{event.original_filename}' in world '{event.world_name}'.")

    try:
        # 1. Calculate hashes from the in-memory stream
        event.file_content.seek(0)
        hashes = hashing_service.calculate_hashes_from_stream(event.file_content, ["sha256", "md5", "crc32", "sha1"])
        sha256_hash = hashes["sha256"]
        logger.debug(f"Calculated sha256 hash: {sha256_hash}")

        # 2. Store the file in the CAS
        event.file_content.seek(0)
        file_bytes = event.file_content.read()
        file_path = await fs_resource.store_file(sha256_hash, file_bytes)
        logger.info(f"Stored file '{event.original_filename}' to CAS at '{file_path}'.")

        # 3. Create and add the File component
        file_component = await ecs_service.add_file_component(
            session=session,
            entity_id=event.entity.id,
            file_path=str(file_path.relative_to(fs_resource.storage_path)),
            size_bytes=len(file_bytes),
            original_filename=event.original_filename,
            sha256=sha256_hash,
            md5=hashes.get("md5"),
            crc32=hashes.get("crc32"),
            sha1=hashes.get("sha1"),
        )
        await session.flush()  # Ensure the file_component gets an ID

        # 4. Fire the FileStored event to continue the pipeline
        file_stored_event = FileStored(
            entity=event.entity,
            file_id=file_component.id,
            file_path=file_path,
        )
        await world.send_event(file_stored_event)
        logger.info(f"Fired FileStored event for entity {event.entity.id}.")

    except Exception as e:
        logger.error(
            f"Failed during ingestion request for entity {event.entity.id} ('{event.original_filename}'): {e}",
            exc_info=True,
        )
        # The scheduler will handle rollback on exception.
        raise


@listens_for(FileStored)
async def asset_dispatcher_system(
    event: FileStored,
    world: Annotated[World, "Resource"],
):
    """
    Listens for when a file has been stored and dispatches it to the
    appropriate processing pipeline based on its MIME type.
    """
    import magic

    logger.info(f"Dispatching asset for entity {event.entity.id} with file path '{event.file_path}'.")

    try:
        mime_type = magic.from_file(str(event.file_path), mime=True)
        logger.info(f"Detected MIME type '{mime_type}' for entity {event.entity.id}.")

        if mime_type.startswith("image/"):
            from dam_media_image.events import ImageAssetDetected

            await world.send_event(ImageAssetDetected(entity=event.entity, file_id=event.file_id))
            logger.info(f"Dispatched entity {event.entity.id} to image processing pipeline.")

        elif mime_type == "application/x-iso9660-image":
            # This is a common MIME type for ISOs.
            # In a real scenario, we might need more robust checking for PSP ISOs.
            from dam_psp.events import PspIsoAssetDetected

            await world.send_event(PspIsoAssetDetected(entity=event.entity, file_id=event.file_id))
            logger.info(f"Dispatched entity {event.entity.id} to PSP ISO processing pipeline.")

        else:
            logger.info(
                f"No specific processing pipeline found for MIME type '{mime_type}' on entity {event.entity.id}."
            )

    except Exception as e:
        logger.error(
            f"Failed during asset dispatch for entity {event.entity.id} ('{event.file_path}'): {e}",
            exc_info=True,
        )
        raise
