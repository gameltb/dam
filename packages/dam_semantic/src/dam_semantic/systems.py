import logging
from typing import Annotated, Dict, List  # Added Annotated

from dam.core.events import AudioSearchQuery, SemanticSearchQuery  # Added AudioSearchQuery
from dam.core.stages import SystemStage  # For scheduling embedding generation
from dam.core.system_params import WorldSession  # Assuming WorldConfig might be needed for model config
from dam.core.systems import listens_for, system
from dam.services import audio_service

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


@system(stage=SystemStage.POST_PROCESSING)
async def generate_embeddings_system(
    session: WorldSession,
):
    """
    Systematically generates text embeddings for entities based on configured text sources.
    This is a simplified version. A more robust implementation would track changes.
    """
    logger.warning("generate_embeddings_system is currently disabled due to removal of ModelExecutionManager.")


@listens_for(SemanticSearchQuery)
async def handle_semantic_search_query(
    event: SemanticSearchQuery,
    session: WorldSession,
):
    """
    Handles a SemanticSearchQuery event, performs the search using the provided ModelExecutionManager,
    and sets the result on the event's future.
    """
    logger.warning("handle_semantic_search_query is currently disabled due to removal of ModelExecutionManager.")
    if event.result_future and not event.result_future.done():
        event.result_future.set_result([])


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
):
    """
    Handles an AudioSearchQuery event, performs the search using AudioService and the provided ModelExecutionManager,
    and sets the result on the event's future.
    """
    logger.warning("handle_audio_search_query is currently disabled due to removal of ModelExecutionManager.")
    if event.result_future and not event.result_future.done():
        event.result_future.set_result([])
