import logging
from typing import Optional

from dam.commands.asset_commands import (
    CheckContentMimeTypeCommand,
    GetMimeTypeCommand,
    RemoveContentMimeTypeCommand,
    SetMimeTypeCommand,
)
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.functions.mime_type_functions import (
    get_content_mime_type,
    set_content_mime_type,
)
from dam.models.metadata.content_mime_type_component import ContentMimeTypeComponent

logger = logging.getLogger(__name__)


@system(on_command=CheckContentMimeTypeCommand)
async def check_content_mime_type_handler(
    cmd: CheckContentMimeTypeCommand,
    transaction: WorldTransaction,
) -> bool:
    """Checks if the ContentMimeTypeComponent exists for the entity."""
    component = await transaction.get_component(cmd.entity_id, ContentMimeTypeComponent)
    return component is not None


@system(on_command=RemoveContentMimeTypeCommand)
async def remove_content_mime_type_handler(
    cmd: RemoveContentMimeTypeCommand,
    transaction: WorldTransaction,
):
    """Removes the ContentMimeTypeComponent from the entity."""
    component = await transaction.get_component(cmd.entity_id, ContentMimeTypeComponent)
    if component:
        await transaction.remove_component(component)
        logger.info(f"Removed ContentMimeTypeComponent from entity {cmd.entity_id}")


@system(on_command=SetMimeTypeCommand)
async def set_mime_type_system(cmd: SetMimeTypeCommand, transaction: WorldTransaction):
    """
    Handles the command to set the mime type for an entity.
    """
    logger.info(f"Setting mime type for entity {cmd.entity_id} to {cmd.mime_type}")
    await set_content_mime_type(transaction.session, cmd.entity_id, cmd.mime_type)


@system(on_command=GetMimeTypeCommand)
async def get_mime_type_system(cmd: GetMimeTypeCommand, transaction: WorldTransaction) -> Optional[str]:
    """
    Handles the command to get the mime type for an entity.
    """
    logger.info(f"Getting mime type for entity {cmd.entity_id}")
    return await get_content_mime_type(transaction.session, cmd.entity_id)
