import logging
import mimetypes
from typing import Annotated

from dam.commands.asset_commands import (
    AutoSetMimeTypeCommand,
    GetAssetFilenamesCommand,
)
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.functions.mime_type_functions import set_entity_mime_type
from dam.models.core.entity import Entity
from dam.models.metadata.mime_type_component import MimeTypeComponent
from sqlalchemy import select

from dam_fs.models.file_properties_component import FilePropertiesComponent

logger = logging.getLogger(__name__)


@system(on_command=AutoSetMimeTypeCommand)
async def auto_set_mime_type_from_filename_system(
    command: AutoSetMimeTypeCommand,
    transaction: EcsTransaction,
    world: Annotated[World, "Resource"],
):
    """
    Automatically sets the mime type for entities with a filename but no mime type.
    """
    if command.entity_id:
        stmt = (
            select(Entity)
            .where(Entity.id == command.entity_id)
            .join(FilePropertiesComponent, Entity.id == FilePropertiesComponent.entity_id)
            .outerjoin(MimeTypeComponent, Entity.id == MimeTypeComponent.entity_id)
            .where(MimeTypeComponent.id.is_(None))
        )
    else:
        stmt = (
            select(Entity)
            .join(FilePropertiesComponent, Entity.id == FilePropertiesComponent.entity_id)
            .outerjoin(MimeTypeComponent, Entity.id == MimeTypeComponent.entity_id)
            .where(MimeTypeComponent.id.is_(None))
        )

    result = await transaction.session.execute(stmt)
    entities_to_process = result.scalars().all()

    if not entities_to_process:
        logger.info("No entities found needing mime type from filename.")
        return

    logger.info(f"Found {len(entities_to_process)} entities to set mime type from filename.")

    for entity in entities_to_process:
        cmd = GetAssetFilenamesCommand(entity_id=entity.id)
        cmd_result = await world.dispatch_command(cmd)

        filenames = list(cmd_result.iter_ok_values_flat())
        if not filenames:
            continue

        sorted_filenames = sorted(filenames, key=lambda f: "." in f, reverse=True)

        mime_type_found = False
        for filename in sorted_filenames:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type:
                logger.info(f"Setting mime type for entity {entity.id} to {mime_type}")
                await set_entity_mime_type(transaction.session, entity.id, mime_type)
                mime_type_found = True
                break

        if not mime_type_found:
            logger.warning(f"Could not guess mime type for any filename for entity {entity.id}")
