import logging
from typing import Optional

from dam.commands.asset_commands import GetMimeTypeCommand, SetMimeTypeCommand
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.functions.mime_type_functions import (
    get_entity_mime_type,
    set_entity_mime_type,
)

logger = logging.getLogger(__name__)


@system(on_command=SetMimeTypeCommand)
async def set_mime_type_system(cmd: SetMimeTypeCommand, transaction: EcsTransaction):
    """
    Handles the command to set the mime type for an entity.
    """
    logger.info(f"Setting mime type for entity {cmd.entity_id} to {cmd.mime_type}")
    await set_entity_mime_type(transaction.session, cmd.entity_id, cmd.mime_type)


@system(on_command=GetMimeTypeCommand)
async def get_mime_type_system(cmd: GetMimeTypeCommand, transaction: EcsTransaction) -> Optional[str]:
    """
    Handles the command to get the mime type for an entity.
    """
    logger.info(f"Getting mime type for entity {cmd.entity_id}")
    return await get_entity_mime_type(transaction.session, cmd.entity_id)
