import logging
from typing import Annotated

import magic
from dam.commands.asset_commands import (
    AutoSetMimeTypeCommand,
    GetAssetStreamCommand,
    SetMimeTypeFromBufferCommand,
)
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.functions.mime_type_functions import set_entity_mime_type
from dam.models.core.entity import Entity
from dam.models.metadata.mime_type_component import MimeTypeComponent
from sqlalchemy import select

from dam_fs.models.filename_component import FilenameComponent

logger = logging.getLogger(__name__)


@system(on_command=AutoSetMimeTypeCommand)
async def auto_set_mime_type_from_filename_system(
    command: AutoSetMimeTypeCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
):
    """
    Automatically sets the mime type for entities by reading the file content.
    """
    if command.entity_id:
        stmt = (
            select(Entity)
            .where(Entity.id == command.entity_id)
            .join(FilenameComponent, Entity.id == FilenameComponent.entity_id)
            .outerjoin(MimeTypeComponent, Entity.id == MimeTypeComponent.entity_id)
            .where(MimeTypeComponent.id.is_(None))
        )
    else:
        stmt = (
            select(Entity)
            .join(FilenameComponent, Entity.id == FilenameComponent.entity_id)
            .outerjoin(MimeTypeComponent, Entity.id == MimeTypeComponent.entity_id)
            .where(MimeTypeComponent.id.is_(None))
        )

    result = await transaction.session.execute(stmt)
    entities_to_process = result.scalars().all()

    if not entities_to_process:
        logger.info("No entities found needing mime type detection.")
        return

    logger.info(f"Found {len(entities_to_process)} entities for mime type detection.")

    for entity in entities_to_process:
        get_stream_cmd = GetAssetStreamCommand(entity_id=entity.id)
        try:
            stream = await world.dispatch_command(get_stream_cmd).get_first_non_none_value()
            if stream:
                with stream:
                    buffer = stream.read(4096)
                    mime_type = magic.from_buffer(buffer, mime=True)
                    if mime_type:
                        logger.info(f"Setting mime type for entity {entity.id} to {mime_type}")
                        await set_entity_mime_type(transaction.session, entity.id, mime_type)
                    else:
                        logger.warning(f"Could not determine mime type for entity {entity.id}")
            else:
                logger.warning(f"Could not get asset stream for entity {entity.id} (no handler returned a stream).")
        except Exception as e:
            logger.error(f"Error processing entity {entity.id}: {e}", exc_info=True)


@system(on_command=SetMimeTypeFromBufferCommand)
async def set_mime_type_from_buffer_system(
    command: SetMimeTypeFromBufferCommand,
    transaction: EcsTransaction,
):
    """
    Sets the mime type for an entity from a buffer, if it doesn't have one.
    """
    existing_mime_type = await transaction.get_component(command.entity_id, MimeTypeComponent)
    if existing_mime_type:
        return

    mime_type = magic.from_buffer(command.buffer, mime=True)
    if mime_type:
        logger.info(f"Setting mime type for entity {command.entity_id} to {mime_type}")
        await set_entity_mime_type(transaction.session, command.entity_id, mime_type)
    else:
        logger.warning(f"Could not determine mime type from buffer for entity {command.entity_id}")
