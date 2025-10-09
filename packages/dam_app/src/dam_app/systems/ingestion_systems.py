"""Defines the asset ingestion and dispatching system."""

import logging
from typing import Annotated

import magic
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.functions.mime_type_functions import set_content_mime_type
from dam_archive.commands import TagArchivePartCommand
from dam_fs.events import FileStored
from dam_media_image.events import ImageAssetDetected

logger = logging.getLogger(__name__)


@system(on_event=FileStored)
async def asset_dispatcher_system(
    event: FileStored,
    world: Annotated[World, "Resource"],
    transaction: WorldTransaction,
):
    """
    Listen for when a file has been stored, store its mime type, and dispatch it.

    The system dispatches the asset to the appropriate processing pipeline
    based on its MIME type.
    """
    logger.info("Dispatching asset for entity %d with file path '%s'.", event.entity.id, event.file_path)

    try:
        mime_type = magic.from_file(str(event.file_path), mime=True)  # type: ignore
        logger.info("Detected MIME type '%s' for entity %d.", mime_type, event.entity.id)

        # Store the mime type using the new refactored function
        await set_content_mime_type(transaction.session, event.entity.id, mime_type)

        if mime_type.startswith("image/"):
            await world.dispatch_event(ImageAssetDetected(entity=event.entity, file_id=event.file_id))
            logger.info("Dispatched entity %d to image processing pipeline.", event.entity.id)

        elif mime_type in [
            "application/zip",
            "application/x-zip-compressed",
            "application/vnd.rar",
            "application/x-rar-compressed",
            "application/x-7z-compressed",
        ]:
            await world.dispatch_command(TagArchivePartCommand(entity_id=event.entity.id))
            logger.info("Dispatched entity %d to archive tagging pipeline.", event.entity.id)

        else:
            logger.info(
                "No specific processing pipeline found for MIME type '%s' on entity %d.", mime_type, event.entity.id
            )

    except Exception as e:
        logger.exception("Failed during asset dispatch for entity %d ('%s'): %s", event.entity.id, event.file_path, e)
        raise
