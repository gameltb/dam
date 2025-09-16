import logging
from typing import Annotated

from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.functions.mime_type_functions import set_content_mime_type
from dam_fs.events import FileStored

logger = logging.getLogger(__name__)


@system(on_event=FileStored)
async def asset_dispatcher_system(
    event: FileStored,
    world: Annotated[World, "Resource"],
    transaction: EcsTransaction,
):
    """
    Listens for when a file has been stored, stores its mime type, and dispatches it
    to the appropriate processing pipeline based on its MIME type.
    """
    import magic

    logger.info(f"Dispatching asset for entity {event.entity.id} with file path '{event.file_path}'.")

    try:
        mime_type = magic.from_file(str(event.file_path), mime=True)  # type: ignore
        logger.info(f"Detected MIME type '{mime_type}' for entity {event.entity.id}.")

        # Store the mime type using the new refactored function
        await set_content_mime_type(transaction.session, event.entity.id, mime_type)

        if mime_type.startswith("image/"):
            from dam_media_image.events import ImageAssetDetected

            await world.dispatch_event(ImageAssetDetected(entity=event.entity, file_id=event.file_id))
            logger.info(f"Dispatched entity {event.entity.id} to image processing pipeline.")

        elif mime_type in [
            "application/zip",
            "application/x-zip-compressed",
            "application/vnd.rar",
            "application/x-rar-compressed",
            "application/x-7z-compressed",
        ]:
            from dam_archive.commands import TagArchivePartCommand

            await world.dispatch_command(TagArchivePartCommand(entity_id=event.entity.id))
            logger.info(f"Dispatched entity {event.entity.id} to archive tagging pipeline.")

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
