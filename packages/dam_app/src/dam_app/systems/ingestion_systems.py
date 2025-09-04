import logging
from typing import Annotated

from dam.core.commands import GetOrCreateEntityFromStreamCommand as CoreGetOrCreateEntityFromStreamCommand
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam_fs.commands import AddFilePropertiesCommand
from dam_fs.events import FileStored
from dam_fs.resources import FileStorageResource

from ..commands import IngestAssetStreamCommand

logger = logging.getLogger(__name__)


@system(on_command=IngestAssetStreamCommand)
async def ingest_asset_stream_command_handler(
    cmd: IngestAssetStreamCommand,
    world: Annotated[World, "Resource"],
    transaction: EcsTransaction,
    fs_resource: Annotated[FileStorageResource, "Resource"],
):
    """
    Handles the command to ingest an in-memory asset stream.
    """
    logger.info(f"Received asset ingestion request for '{cmd.original_filename}' in world '{cmd.world_name}'.")

    try:
        cmd.file_content.seek(0)
        size_bytes = len(cmd.file_content.getvalue())

        # 1. Get or create entity from stream
        get_or_create_cmd = CoreGetOrCreateEntityFromStreamCommand(
            stream=cmd.file_content,
        )
        command_result = await world.dispatch_command(get_or_create_cmd)
        entity, sha256_bytes = command_result.get_one_value()

        # 2. Add file properties
        add_props_cmd = AddFilePropertiesCommand(
            entity_id=entity.id,
            original_filename=cmd.original_filename,
            size_bytes=size_bytes,
        )
        await world.dispatch_command(add_props_cmd)

        await transaction.flush()

        # TODO: The FileStored event needs a file_id, which is not directly available
        # from the import_stream function. This needs to be resolved.
        # For now, we will not fire the event.
        logger.info(
            f"Successfully processed IngestAssetStreamCommand for {cmd.original_filename}, created Entity {entity.id}"
        )

    except Exception as e:
        logger.error(
            f"Failed during ingestion request for entity {cmd.entity.id} ('{cmd.original_filename}'): {e}",
            exc_info=True,
        )
        # The scheduler will handle rollback on exception.
        raise


@system(on_event=FileStored)
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
