"""Defines systems for automatically setting MIME types."""

import logging
from typing import Annotated

import magic
from dam.commands.analysis_commands import AutoSetMimeTypeCommand
from dam.commands.asset_commands import GetAssetFilenamesCommand, SetMimeTypeFromBufferCommand
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.functions.mime_type_functions import set_content_mime_type
from dam.models.metadata.content_mime_type_component import ContentMimeTypeComponent

logger = logging.getLogger(__name__)

EXTENSION_TO_MIMETYPE = {
    ".zip": "application/zip",
    ".rar": "application/vnd.rar",
    ".7z": "application/x-7z-compressed",
}


@system(on_command=AutoSetMimeTypeCommand)
async def auto_set_mime_type_from_filename_system(
    command: AutoSetMimeTypeCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
):
    """Automatically sets the mime type for entities by reading the file content."""
    entity_id = command.entity_id

    # Check if mime type is already set
    existing_mime_type = await transaction.get_component(entity_id, ContentMimeTypeComponent)
    if existing_mime_type:
        return

    try:
        provider = await command.get_stream_provider(world)
        if not provider:
            return

        async with provider.get_stream() as stream:
            buffer = stream.read(4096)
            mime_type = magic.from_buffer(buffer, mime=True)

            if mime_type == "application/octet-stream":
                # Fallback to file extension
                filenames = await world.dispatch_command(
                    GetAssetFilenamesCommand(entity_id=entity_id)
                ).get_first_non_none_value()
                if filenames:
                    for filename in filenames:
                        for ext, mt in EXTENSION_TO_MIMETYPE.items():
                            if filename.endswith(ext):
                                mime_type = mt
                                break
                        if mime_type != "application/octet-stream":
                            break

            if mime_type:
                logger.info("Setting mime type for entity %s to %s", entity_id, mime_type)
                await set_content_mime_type(transaction.session, entity_id, mime_type)
            else:
                logger.warning("Could not determine mime type for entity %s", entity_id)
    except Exception:
        logger.exception("Error processing entity %s", entity_id)


@system(on_command=SetMimeTypeFromBufferCommand)
async def set_mime_type_from_buffer_system(
    command: SetMimeTypeFromBufferCommand,
    transaction: WorldTransaction,
) -> None:
    """Set the mime type for an entity from a buffer, if it doesn't have one."""
    existing_mime_type = await transaction.get_component(command.entity_id, ContentMimeTypeComponent)
    if existing_mime_type:
        return

    mime_type = magic.from_buffer(command.buffer, mime=True)
    if mime_type:
        logger.info("Setting mime type for entity %s to %s", command.entity_id, mime_type)
        await set_content_mime_type(transaction.session, command.entity_id, mime_type)
    else:
        logger.warning("Could not determine mime type from buffer for entity %s", command.entity_id)
