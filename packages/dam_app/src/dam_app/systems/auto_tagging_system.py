import logging
from typing import Annotated, List

from dam.core.system_params import (
    WorldContext,
)

# For getting session, world name, config (WorldContext is in system_params)
from dam.core.systems import SystemStage, system
from dam.models.core.entity import Entity  # Corrected Entity import
from dam.services import ecs_service

logger = logging.getLogger(__name__)


from dam.core.components_markers import NeedsAutoTaggingMarker


@system(stage=SystemStage.CONTENT_ANALYSIS)
async def auto_tag_entities_system(
    world_context: Annotated[WorldContext, "WorldContext"],
    marked_entities: Annotated[List[Entity], "MarkedEntityList", NeedsAutoTaggingMarker],
):
    """
    System that processes entities marked with NeedsAutoTaggingMarker,
    generates tags using a configured model, and stores them.
    """
    session = world_context.session
    if not marked_entities:
        logger.debug("No entities marked for auto-tagging in this cycle.")
        return
    logger.warning("Auto-tagging system is currently disabled due to removal of ModelExecutionManager.")
    for entity in marked_entities:
        marker = await ecs_service.get_component(session, entity.id, NeedsAutoTaggingMarker)
        if marker:
            await ecs_service.remove_component(session, marker)

    # Session flush/commit is typically handled by the WorldScheduler after all systems in a stage run.
    # If immediate commit is needed for some reason (rare for systems), it should be done carefully.
    # await session.commit() # Usually not done here.


# To make this system runnable, it needs to be registered with the WorldScheduler,
# typically in world_setup.py or a similar central setup location.
# Also, NeedsAutoTaggingMarker and AutoTaggingCompleteMarker need to be registered
# component types if they are to be used by ecs_service.add_component etc.
# (This is often done by importing them in dam.core.components_markers or a similar file
# that's imported early, or by explicit registration calls).
# For now, assume they are standard components.
