import logging
from typing import Annotated

from dam.core.system_params import (
    WorldContext,
)

# For getting session, world name, config (WorldContext is in system_params)
from dam.core.systems import handles_command

from ..commands import AutoTagEntityCommand

logger = logging.getLogger(__name__)


from dam_sire.resource import SireResource


@handles_command(AutoTagEntityCommand)
async def auto_tag_entity_command_handler(
    cmd: AutoTagEntityCommand,
    world_context: Annotated[WorldContext, "WorldContext"],
    sire_resource: Annotated[SireResource, "Resource"],
):
    """
    Handles the command to auto-tag a single entity.
    """
    session = world_context.session
    entity = cmd.entity
    logger.info(f"Handling AutoTagEntityCommand for entity {entity.id}")

    from dam.utils.media_utils import get_file_path_for_entity

    from dam_app.functions import tagging_functions as tagging_functions_module

    image_path = await get_file_path_for_entity(session, entity.id, world_context.world_config.ASSET_STORAGE_PATH)
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

    # Session flush/commit is typically handled by the WorldScheduler after all systems in a stage run.
    # If immediate commit is needed for some reason (rare for systems), it should be done carefully.
    # await session.commit() # Usually not done here.


# To make this system runnable, it needs to be registered with the WorldScheduler,
# typically in world_setup.py or a similar central setup location.
# Also, NeedsAutoTaggingMarker and AutoTaggingCompleteMarker need to be registered
# component types if they are to be used by ecs_functions.add_component etc.
# (This is often done by importing them in dam.core.components_markers or a similar file
# that's imported early, or by explicit registration calls).
# For now, assume they are standard components.
