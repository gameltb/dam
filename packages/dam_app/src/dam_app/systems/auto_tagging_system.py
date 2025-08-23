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
from dam_sire.resource import SireResource


@system(stage=SystemStage.CONTENT_ANALYSIS)
async def auto_tag_entities_system(
    world_context: Annotated[WorldContext, "WorldContext"],
    sire_resource: Annotated[SireResource, "Resource"],
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

    from dam.utils.media_utils import get_file_path_for_entity

    from dam_app.services import tagging_service as tagging_service_module

    for entity in marked_entities:
        marker = await ecs_service.get_component(session, entity.id, NeedsAutoTaggingMarker)
        if not marker:
            continue

        image_path = await get_file_path_for_entity(session, entity.id, world_context.world_config.ASSET_STORAGE_PATH)
        if not image_path:
            logger.warning(f"Could not determine image file path for entity {entity.id}. Skipping auto-tagging.")
            await ecs_service.remove_component(session, marker)
            continue

        try:
            await tagging_service_module.update_entity_model_tags(
                session,
                sire_resource,
                entity.id,
                str(image_path),
                "wd-v1-4-moat-tagger-v2",
            )
            await ecs_service.remove_component(session, marker)
        except Exception as e:
            logger.error(f"Error auto-tagging entity {entity.id}: {e}", exc_info=True)

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
