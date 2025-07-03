import logging
from typing import List, Annotated

from sqlalchemy.ext.asyncio import AsyncSession

from dam.core.config import AppConfig
from dam.core.core_types import WorldContext # For getting session, world name, config
from dam.core.entity import Entity
from dam.core.components_markers import MarkEntityForProcessingComponent, ProcessingScope
from dam.core.systems import system, SystemStage
from dam.services.tagging_service import TaggingService # Assuming tagging_service is in dam.services
from dam.models.core.file_location_component import FileLocationComponent # To get image path

logger = logging.getLogger(__name__)

# Define a specific marker for auto-tagging
class NeedsAutoTaggingMarker(MarkEntityForProcessingComponent):
    processing_scope: ProcessingScope = ProcessingScope.SINGLE_ENTITY
    # Add any specific fields if needed for this marker, e.g., model_to_use
    # model_name_to_apply: str = "wd-v1-4-moat-tagger-v2" # Could be a field in the marker

class AutoTaggingCompleteMarker(MarkEntityForProcessingComponent):
    processing_scope: ProcessingScope = ProcessingScope.SINGLE_ENTITY
    # model_name_applied: str # Store which model was applied

@system(stage=SystemStage.PROCESSING, depends_on_resources=[TaggingService])
async def auto_tag_entities_system(
    world_context: Annotated[WorldContext, "WorldContext"],
    marked_entities: Annotated[List[Entity], "MarkedEntityList", NeedsAutoTaggingMarker],
    tagging_service: Annotated[TaggingService, "Resource"], # Injected TaggingService
):
    """
    System that processes entities marked with NeedsAutoTaggingMarker,
    generates tags using a configured model, and stores them.
    """
    session = world_context.session
    config = world_context.config # Access AppConfig if needed for model names, etc.

    # Example: Get default model name from config or use a fixed one
    # default_tagging_model = config.get("DEFAULT_AUTO_TAGGING_MODEL", "wd-v1-4-moat-tagger-v2")
    default_tagging_model = "wd-v1-4-moat-tagger-v2" # Hardcoded for now

    if not marked_entities:
        logger.debug("No entities marked for auto-tagging in this cycle.")
        return

    logger.info(f"Found {len(marked_entities)} entities marked for auto-tagging with model '{default_tagging_model}'.")

    for entity in marked_entities:
        marker = await world_context.ecs.get_component(session, entity.id, NeedsAutoTaggingMarker)
        if not marker: # Should not happen if marked_entities list is correct
            continue

        # model_to_use = getattr(marker, 'model_name_to_apply', default_tagging_model)
        model_to_use = default_tagging_model # Using the default for now

        # Get the image path for the entity.
        # This assumes the entity has a FileLocationComponent pointing to an image.
        # You might need more sophisticated logic to find the "primary" image of an entity.
        file_location_comp = await world_context.ecs.get_component(session, entity.id, FileLocationComponent)
        if not file_location_comp or not file_location_comp.full_path:
            logger.warning(f"Entity {entity.id} marked for auto-tagging has no FileLocationComponent or path. Skipping.")
            # Optionally, remove marker or add error marker
            await world_context.ecs.remove_component_from_entity(session, entity.id, NeedsAutoTaggingMarker)
            continue

        image_path = file_location_comp.full_path
        # Ensure this path is accessible by the tagging model.
        # If models run in different containers/environments, path translation might be needed.

        logger.info(f"Processing entity {entity.id} for auto-tagging with model '{model_to_use}' using image: {image_path}")

        try:
            await tagging_service.update_entity_model_tags(
                session,
                entity.id,
                image_path, # Pass the image path
                model_to_use
            )
            logger.info(f"Successfully applied tags from model '{model_to_use}' to entity {entity.id}.")

            # Remove the NeedsAutoTaggingMarker
            await world_context.ecs.remove_component_from_entity(session, entity.id, NeedsAutoTaggingMarker)

            # Add AutoTaggingCompleteMarker (optional, for tracking)
            # completion_marker = AutoTaggingCompleteMarker(model_name_applied=model_to_use)
            # await world_context.ecs.add_component_to_entity(session, entity.id, completion_marker)

        except Exception as e:
            logger.error(f"Error auto-tagging entity {entity.id} with model '{model_to_use}': {e}", exc_info=True)
            # Optionally, add an error marker component to the entity
            # await world_context.ecs.remove_component_from_entity(session, entity.id, NeedsAutoTaggingMarker) # Remove to avoid reprocessing loop on same error
            # error_marker = AutoTaggingErrorMarker(model_name=model_to_use, error_message=str(e))
            # await world_context.ecs.add_component_to_entity(session, entity.id, error_marker)

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
