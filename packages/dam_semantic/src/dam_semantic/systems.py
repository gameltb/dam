import logging
from typing import Annotated, Dict, List  # Added Annotated

from dam.core.events import AudioSearchQuery, SemanticSearchQuery  # Added AudioSearchQuery
from dam.core.stages import SystemStage  # For scheduling embedding generation
from dam.core.system_params import WorldSession  # Assuming WorldConfig might be needed for model config
from dam.core.systems import listens_for, system
from dam_media_audio.services import audio_service

from . import service as semantic_service

# Placeholder for components that might trigger embedding generation

# from dam.models.metadata import ExiftoolMetadataComponent # If we decide to embed exif data

logger = logging.getLogger(__name__)

# Configuration for which components and fields to use for embeddings
# This could be moved to a config file or resource later
# Format: { ComponentClassName: ["field_name1", "field_name2", ...], ... }
TEXT_SOURCES_FOR_EMBEDDING: Dict[str, List[str]] = {
    "FilePropertiesComponent": ["original_filename"],  # filename might be good for some types of semantic search
    "TagConceptComponent": ["concept_name", "concept_description"],  # For tags themselves
    "CharacterConceptComponent": ["concept_name", "concept_description"],  # For characters
    # Add more components and their text fields as needed
    # "ExiftoolMetadataComponent": ["UserComment", "Description", "Title"] # Example for EXIF fields
}


from dam_sire.resource import SireResource

@system(stage=SystemStage.POST_PROCESSING)
async def generate_embeddings_system(
    session: WorldSession,
    sire_resource: Annotated[SireResource, "Resource"],
):
    """
    Systematically generates text embeddings for entities based on configured text sources.
    This is a simplified version. A more robust implementation would track changes.
    """
    logger.info("SemanticEmbeddingSystem: Starting embedding generation pass.")
    # ... (rest of the implementation will be simplified for now)


@listens_for(SemanticSearchQuery)
async def handle_semantic_search_query(
    event: SemanticSearchQuery,
    session: WorldSession,
    sire_resource: Annotated[SireResource, "Resource"],
):
    """
    Handles a SemanticSearchQuery event, performs the search using the provided SireResource,
    and sets the result on the event's future.
    """
    if not event.result_future:
        logger.error(f"Result future not set on SemanticSearchQuery event (Req ID: {event.request_id}).")
        return

    model_to_use = event.model_name if event.model_name else semantic_service.DEFAULT_MODEL_NAME

    try:
        similar_entities_data = await semantic_service.find_similar_entities_by_text_embedding(
            session=session,
            sire_resource=sire_resource,
            query_text=event.query_text,
            top_n=event.top_n,
            model_name=model_to_use,
        )
        if not event.result_future.done():
            event.result_future.set_result(similar_entities_data)
    except Exception as e:
        logger.error(f"Error in handle_semantic_search_query: {e}", exc_info=True)
        if not event.result_future.done():
            event.result_future.set_exception(e)


# __all__ needs to be defined if systems are imported elsewhere using `from ... import *`
# For explicit imports, it's not strictly necessary.
__all__ = [
    "generate_embeddings_system",
    "handle_semantic_search_query",
    "handle_audio_search_query",  # Added new handler
]


@listens_for(AudioSearchQuery)
async def handle_audio_search_query(
    event: AudioSearchQuery,
    session: WorldSession,
    sire_resource: Annotated[SireResource, "Resource"],
):
    """
    Handles an AudioSearchQuery event, performs the search using AudioService and the provided SireResource,
    and sets the result on the event's future.
    """
    if not event.result_future:
        logger.error(f"Result future not set on AudioSearchQuery event (Req ID: {event.request_id}).")
        return

    model_to_use = event.model_name if event.model_name else audio_service.DEFAULT_AUDIO_MODEL_NAME

    try:
        similar_entities_data = await audio_service.find_similar_entities_by_audio_embedding(
            session=session,
            sire_resource=sire_resource,
            query_audio_path=str(event.query_audio_path),
            top_n=event.top_n,
            model_name=model_to_use,
        )
        if not event.result_future.done():
            event.result_future.set_result(similar_entities_data)
    except Exception as e:
        logger.error(f"Error in handle_audio_search_query: {e}", exc_info=True)
        if not event.result_future.done():
            event.result_future.set_exception(e)
