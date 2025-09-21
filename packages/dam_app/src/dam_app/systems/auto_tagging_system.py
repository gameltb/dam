import logging
from typing import Annotated

from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam_sire.resource import SireResource

from ..commands import AutoTagEntityCommand

logger = logging.getLogger(__name__)


@system(on_command=AutoTagEntityCommand)
async def auto_tag_entity_command_handler(
    cmd: AutoTagEntityCommand,
    transaction: WorldTransaction,
    world: Annotated[World, "Resource"],
    sire_resource: SireResource,
):
    """
    Handles the command to auto-tag a single entity.
    """
    session = transaction.session
    entity = cmd.entity
    logger.info(f"Handling AutoTagEntityCommand for entity {entity.id}")

    from dam_fs.functions import file_operations as file_operations_module

    from dam_app.functions import tagging_functions as tagging_functions_module

    image_path = await file_operations_module.get_file_path_for_entity(world, transaction, entity.id)
    if not image_path:
        logger.warning(f"Could not determine image file path for entity {entity.id}. Skipping auto-tagging.")
        return

    try:
        await tagging_functions_module.update_entity_model_tags(
            session,
            sire_resource,
            entity.id,
            str(image_path),
            "wd-v1-4-moat-tagger-v2",
        )
    except Exception as e:
        logger.error(f"Error auto-tagging entity {entity.id}: {e}", exc_info=True)
