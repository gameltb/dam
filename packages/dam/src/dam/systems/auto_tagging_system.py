import logging
from typing import Annotated, List

from dam.core.model_manager import ModelExecutionManager  # Added
from dam.core.system_params import (
    WorldContext,
)

# For getting session, world name, config (WorldContext is in system_params)
from dam.core.systems import SystemStage, system
from dam.models.core.base_component import BaseComponent  # Import BaseComponent directly
from dam.models.core.entity import Entity  # Corrected Entity import
from dam.services import ecs_service

logger = logging.getLogger(__name__)


from dam.core.components_markers import NeedsAutoTaggingMarker, AutoTaggingCompleteMarker


@system(stage=SystemStage.CONTENT_ANALYSIS)  # Removed depends_on_resources
async def auto_tag_entities_system(
    world_context: Annotated[WorldContext, "WorldContext"],
    model_execution_manager: Annotated[ModelExecutionManager, "Resource"],  # Changed to inject MEM
    marked_entities: Annotated[List[Entity], "MarkedEntityList", NeedsAutoTaggingMarker],
):
    """
    System that processes entities marked with NeedsAutoTaggingMarker,
    generates tags using a configured model, and stores them.
    """
    session = world_context.session
    config = world_context.config  # Access AppConfig if needed for model names, etc.

    # Example: Get default model name from config or use a fixed one
    # default_tagging_model = config.get("DEFAULT_AUTO_TAGGING_MODEL", "wd-v1-4-moat-tagger-v2")
    default_tagging_model = "wd-v1-4-moat-tagger-v2"  # Hardcoded for now

    if not marked_entities:
        logger.debug("No entities marked for auto-tagging in this cycle.")
        return

    from dam.services import tagging_service as tagging_service_module

    logger.info(f"Found {len(marked_entities)} entities marked for auto-tagging with model '{default_tagging_model}'.")

    for entity in marked_entities:
        # Corrected: Use ecs_service directly
        marker = await ecs_service.get_component(session, entity.id, NeedsAutoTaggingMarker)
        if not marker:  # Should not happen if marked_entities list is correct
            continue

        # model_to_use = getattr(marker, 'model_name_to_apply', default_tagging_model)
        model_to_use = default_tagging_model  # Using the default for now

        # Get the image path for the entity.
        # Use the get_file_path_for_entity utility
        from dam.utils.media_utils import get_file_path_for_entity

        image_path = await get_file_path_for_entity(session, entity.id, world_context.world_config.ASSET_STORAGE_PATH)

        if not image_path:
            logger.warning(
                f"Entity {entity.id} marked for auto-tagging: Could not determine image file path. Skipping."
            )
            await ecs_service.remove_component(session, marker)  # Pass the marker instance to remove
            continue
        # Ensure this path is accessible by the tagging model.
        # If models run in different containers/environments, path translation might be needed.

        logger.info(
            f"Processing entity {entity.id} for auto-tagging with model '{model_to_use}' using image: {image_path}"
        )

        try:
            await tagging_service_module.update_entity_model_tags(  # Call module function
                session,
                model_execution_manager,  # Pass MEM
                entity.id,
                image_path,
                model_to_use,
            )
            logger.info(f"Successfully applied tags from model '{model_to_use}' to entity {entity.id}.")

            # Remove the NeedsAutoTaggingMarker by passing the marker instance
            await ecs_service.remove_component(session, marker)

            # Add AutoTaggingCompleteMarker (optional, for tracking)
            # completion_marker = AutoTaggingCompleteMarker(model_name_applied=model_to_use) # Create instance
            # await ecs_service.add_component_to_entity(session, entity.id, completion_marker) # Use ecs_service

        except Exception as e:
            logger.error(f"Error auto-tagging entity {entity.id} with model '{model_to_use}': {e}", exc_info=True)
            # Optionally, add an error marker component to the entity
            # await ecs_service.remove_component(session, marker) # Remove to avoid reprocessing loop
            # error_marker = AutoTaggingErrorMarker(model_name=model_to_use, error_message=str(e)) # Create instance
            # await ecs_service.add_component_to_entity(session, entity.id, error_marker) # Use ecs_service

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
