import logging
from typing import Annotated, Dict, List  # Added Annotated

from dam.core.stages import SystemStage
from dam.core.systems import handles_command, system
from dam.core.transaction import EcsTransaction
from dam_media_audio.commands import AudioSearchCommand
from dam_media_audio.services import audio_service

from . import service as semantic_service
from .commands import SemanticSearchCommand

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
    transaction: EcsTransaction,
    sire_resource: Annotated[SireResource, "Resource"],
):
    """
    Systematically generates text embeddings for entities based on configured text sources.
    This is a simplified version. A more robust implementation would track changes.
    """
    logger.info("SemanticEmbeddingSystem: Starting embedding generation pass.")
    # ... (rest of the implementation will be simplified for now)


@handles_command(SemanticSearchCommand)
async def handle_semantic_search_command(
    cmd: SemanticSearchCommand,
    transaction: EcsTransaction,
    sire_resource: Annotated[SireResource, "Resource"],
):
    """
    Handles a SemanticSearchCommand, performs the search using the provided SireResource,
    and sets the result on the command's future.
    """
    if not cmd.result_future:
        logger.error(f"Result future not set on SemanticSearchCommand (Req ID: {cmd.request_id}).")
        return

    model_to_use = cmd.model_name if cmd.model_name else semantic_service.DEFAULT_MODEL_NAME

    try:
        similar_entities_data = await semantic_service.find_similar_entities_by_text_embedding(
            transaction=transaction,
            sire_resource=sire_resource,
            query_text=cmd.query_text,
            top_n=cmd.top_n,
            model_name=model_to_use,
        )
        if not cmd.result_future.done():
            cmd.result_future.set_result(similar_entities_data)
    except Exception as e:
        logger.error(f"Error in handle_semantic_search_command: {e}", exc_info=True)
        if not cmd.result_future.done():
            cmd.result_future.set_exception(e)


# __all__ needs to be defined if systems are imported elsewhere using `from ... import *`
# For explicit imports, it's not strictly necessary.
__all__ = [
    "generate_embeddings_system",
    "handle_semantic_search_command",
    "handle_audio_search_command",
]


@handles_command(AudioSearchCommand)
async def handle_audio_search_command(
    cmd: AudioSearchCommand,
    transaction: EcsTransaction,
    sire_resource: Annotated[SireResource, "Resource"],
):
    """
    Handles an AudioSearchCommand, performs the search using AudioService and the provided SireResource,
    and sets the result on the command's future.
    """
    if not cmd.result_future:
        logger.error(f"Result future not set on AudioSearchCommand (Req ID: {cmd.request_id}).")
        return

    model_to_use = cmd.model_name if cmd.model_name else audio_service.DEFAULT_AUDIO_MODEL_NAME

    try:
        similar_entities_data = await audio_service.find_similar_entities_by_audio_embedding(
            transaction=transaction,
            sire_resource=sire_resource,
            query_audio_path=str(cmd.query_audio_path),
            top_n=cmd.top_n,
            model_name=model_to_use,
        )
        if not cmd.result_future.done():
            cmd.result_future.set_result(similar_entities_data)
    except Exception as e:
        logger.error(f"Error in handle_audio_search_command: {e}", exc_info=True)
        if not cmd.result_future.done():
            cmd.result_future.set_exception(e)
