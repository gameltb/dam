import logging
from typing import List, Dict, Any, Optional, Tuple, Type

from dam.core.events import SemanticSearchQuery
from dam.core.systems import listens_for, system
from dam.core.system_params import WorldSession, WorldConfig # Assuming WorldConfig might be needed for model config
from dam.core.stages import SystemStage # For scheduling embedding generation

from dam.services import semantic_service, ecs_service
from dam.models.core.entity import Entity
from dam.models.semantic import TextEmbeddingComponent

# Placeholder for components that might trigger embedding generation
from dam.models.properties import FilePropertiesComponent
from dam.models.conceptual import TagConceptComponent, CharacterConceptComponent, EntityTagLinkComponent
# from dam.models.metadata import ExiftoolMetadataComponent # If we decide to embed exif data

logger = logging.getLogger(__name__)

# Configuration for which components and fields to use for embeddings
# This could be moved to a config file or resource later
# Format: { ComponentClassName: ["field_name1", "field_name2", ...], ... }
TEXT_SOURCES_FOR_EMBEDDING: Dict[str, List[str]] = {
    "FilePropertiesComponent": ["original_filename"], # filename might be good for some types of semantic search
    "TagConceptComponent": ["concept_name", "concept_description"], # For tags themselves
    "CharacterConceptComponent": ["concept_name", "concept_description"], # For characters
    # Add more components and their text fields as needed
    # "ExiftoolMetadataComponent": ["UserComment", "Description", "Title"] # Example for EXIF fields
}


@system(stage=SystemStage.POST_PROCESSING) # Or a new SEMANTIC_PROCESSING stage
async def generate_embeddings_system(
    session: WorldSession,
    # world_config: WorldConfig, # If model name or other settings come from world config
    # For now, using default model from semantic_service
    # We need a way to get entities that were created/updated in the current transaction/tick
    # This is a common ECS challenge. Options:
    # 1. Listen for specific "EntityCreated", "ComponentAdded<T>" events (more complex event system).
    # 2. Have a marker component like "NeedsEmbeddingUpdate" added by other systems.
    # 3. Query for all entities and check an "updated_at" timestamp (if models have it).
    # For simplicity, let's assume for now this system runs periodically or is manually triggered
    # on a selection of entities, or it queries for entities with relevant components
    # that don't yet have embeddings for the current default model.

    # Let's try a simple approach: find entities with text source components that lack embeddings.
    # This won't handle updates to existing text fields well without more state.
):
    """
    Systematically generates text embeddings for entities based on configured text sources.
    This is a simplified version. A more robust implementation would track changes.
    """
    logger.info("SemanticEmbeddingSystem: Starting embedding generation pass.")
    model_name_to_use = semantic_service.DEFAULT_MODEL_NAME # Or from world_config

    entities_processed_count = 0
    embeddings_created_count = 0

    for comp_name_str, field_names in TEXT_SOURCES_FOR_EMBEDDING.items():
        # Dynamically get component class - requires component registration to be complete
        # This is a bit tricky. For now, let's hardcode checks or assume comp_name_str matches class name.
        # A better way would be to register component classes with their names.

        # Simplified: Iterate through all entities that have one of the source components
        # This is inefficient for large DBs. A marker component "NeedsEmbedding" would be better.

        # Find all entities that have the source component
        # This needs ecs_service.find_entities_with_component_class_name(comp_name_str) or similar
        # For now, let's imagine we get a list of relevant entities.
        # This is a placeholder for a more efficient entity selection strategy.

        # Example: Iterate over all entities and check for components (very inefficient)
        # More realistically, this system would react to EntityUpdated events or NeedsEmbeddingUpdate markers

        # Let's refine: query for entities that have a source component BUT NOT a corresponding embedding
        # This is still not perfect as it doesn't handle updates to source text.

        # For TagConceptComponent and CharacterConceptComponent, these are "conceptual entities"
        # themselves, so we'd iterate over entities having these components.
        # For FilePropertiesComponent, it's on asset entities.

        # This part needs a more robust way to identify entities needing embedding updates.
        # For now, as a conceptual placeholder, let's assume we iterate relevant entities:

        # Let's assume we are processing entities that have been marked (e.g. by an ingestion system)
        # with a (hypothetical) "NeedsSemanticProcessing" marker.
        # For this example, let's just process ALL entities that have ANY of the source components
        # This is for demonstration and would be too slow in practice.

        # A slightly better approach for this pass:
        # For each component type in TEXT_SOURCES_FOR_EMBEDDING:
        #   Get all entities with that component.
        #   For each of these entities:
        #     Check if they have an embedding for that component.field and model.
        #     If not, generate it.
        # This is still not handling updates to the source text.

        # This system's trigger and entity selection mechanism needs careful design in a full ECS.
        # For now, this system will be manually invoked or triggered by a broad event.
        # To make it runnable, let's assume it just processes a small number of entities for now,
        # or we'd need a way to get "recently modified" entities.

        # For the purpose of this step, the system structure is the focus.
        # The actual logic for selecting entities to process will be refined later if needed.
        pass # Placeholder for entity iteration logic

    # logger.info(f"SemanticEmbeddingSystem: Finished. Processed {entities_processed_count} entities, created/updated {embeddings_created_count} embeddings.")
    # Actual implementation of entity iteration and text gathering would go here.
    # Example for a single entity (if entity_id was passed in an event):
    # entity_id = ...
    # texts_to_batch_for_entity = []
    # for comp_class_name, field_names_list in TEXT_SOURCES_FOR_EMBEDDING.items():
    #     comp_class = ecs_service.get_registered_component_type_by_name(comp_class_name)
    #     if not comp_class: continue
    #     component = await ecs_service.get_component(session, entity_id, comp_class)
    #     if component:
    #         for field_name in field_names_list:
    #             if hasattr(component, field_name):
    #                 text_val = getattr(component, field_name)
    #                 if isinstance(text_val, str) and text_val.strip():
    #                     texts_to_batch_for_entity.append( (comp_class_name, field_name, text_val) )
    # if texts_to_batch_for_entity:
    #    await semantic_service.update_text_embeddings_for_entity(session, entity_id, {}, model_name_to_use, batch_texts=texts_to_batch_for_entity)


@listens_for(SemanticSearchQuery)
async def handle_semantic_search_query(
    event: SemanticSearchQuery,
    session: WorldSession,
    # world_config: WorldConfig, # If model name could come from world config
):
    """
    Handles a SemanticSearchQuery event, performs the search, and sets the result on the event's future.
    """
    # Ensure this log appears to confirm entry
    logger.critical(f"CRITICAL_LOG: ENTERING handle_semantic_search_query for Req ID: {event.request_id}")

    logger.info(
        f"SemanticSearchSystem: Handling SemanticSearchQuery (Req ID: {event.request_id}) "
        f"for query: '{event.query_text[:50]}...' in world '{event.world_name}'"
    )

    if not event.result_future:
        logger.error(f"Result future not set on SemanticSearchQuery event (Req ID: {event.request_id}). Cannot proceed.")
        return

    model_to_use = event.model_name if event.model_name else semantic_service.DEFAULT_MODEL_NAME

    try:
        # The service function returns List[Tuple[Entity, float, TextEmbeddingComponent]]
        # The future expects List[Tuple[Any, float, Any]] to avoid model imports in events.py
        similar_entities_data = await semantic_service.find_similar_entities_by_text_embedding(
            session=session,
            query_text=event.query_text,
            top_n=event.top_n,
            model_name=model_to_use,
        )

        # Convert Entity and TextEmbeddingComponent to a serializable form if needed,
        # or ensure the CLI/UI can handle the direct objects.
        # For now, assume direct objects are fine for the future.
        # The type hint in SemanticSearchQuery.result_future is List[Tuple[Any, float, Any]]
        # which matches the structure.

        if not event.result_future.done():
            event.result_future.set_result(similar_entities_data)
        logger.info(f"SemanticSearchSystem: Query (Req ID: {event.request_id}) completed. Found {len(similar_entities_data)} results.")

    except Exception as e:
        logger.error(f"Error in handle_semantic_search_query (Req ID: {event.request_id}): {e}", exc_info=True)
        if not event.result_future.done():
            event.result_future.set_exception(e)

# __all__ needs to be defined if systems are imported elsewhere using `from ... import *`
# For explicit imports, it's not strictly necessary.
__all__ = [
    "generate_embeddings_system",
    "handle_semantic_search_query",
]
