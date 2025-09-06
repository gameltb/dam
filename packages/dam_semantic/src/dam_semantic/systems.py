import logging
from typing import Annotated, Any, Dict, List, Optional, Tuple

from dam.core.stages import SystemStage
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam_media_audio.commands import AudioSearchCommand
from dam_media_audio.functions import audio_functions
from dam_sire.resource import SireResource

from . import semantic_functions
from .commands import SemanticSearchCommand

logger = logging.getLogger(__name__)

TEXT_SOURCES_FOR_EMBEDDING: Dict[str, List[str]] = {
    "FilePropertiesComponent": ["original_filename"],
    "TagConceptComponent": ["concept_name", "concept_description"],
    "CharacterConceptComponent": ["concept_name", "concept_description"],
}


@system(stage=SystemStage.POST_PROCESSING)
async def generate_embeddings_system(
    transaction: EcsTransaction,
    sire_resource: Annotated[SireResource, "Resource"],
) -> None:
    logger.info("SemanticEmbeddingSystem: Starting embedding generation pass.")


@system(on_command=SemanticSearchCommand)
async def handle_semantic_search_command(
    cmd: SemanticSearchCommand,
    transaction: EcsTransaction,
    sire_resource: Annotated[SireResource, "Resource"],
) -> Optional[List[Tuple[Any, float, Any]]]:
    model_to_use = cmd.model_name if cmd.model_name else semantic_functions.DEFAULT_MODEL_NAME

    try:
        similar_entities_data = await semantic_functions.find_similar_entities_by_text_embedding(
            transaction=transaction,
            sire_resource=sire_resource,
            query_text=cmd.query_text,
            top_n=cmd.top_n,
            model_name=model_to_use,
        )
        return similar_entities_data
    except Exception as e:
        logger.error(f"Error in handle_semantic_search_command: {e}", exc_info=True)
        raise


@system(on_command=AudioSearchCommand)
async def handle_audio_search_command(
    cmd: AudioSearchCommand,
    transaction: EcsTransaction,
    sire_resource: Annotated[SireResource, "Resource"],
) -> Optional[List[Tuple[Any, float, Any]]]:
    model_to_use = cmd.model_name if cmd.model_name else audio_functions.DEFAULT_AUDIO_MODEL_NAME

    try:
        similar_entities_data = await audio_functions.find_similar_entities_by_audio_embedding(
            transaction=transaction,
            sire_resource=sire_resource,
            query_audio_path=str(cmd.query_audio_path),
            top_n=cmd.top_n,
            model_name=model_to_use,
        )
        return similar_entities_data
    except Exception as e:
        logger.error(f"Error in handle_audio_search_command: {e}", exc_info=True)
        raise


__all__ = [
    "generate_embeddings_system",
    "handle_semantic_search_command",
    "handle_audio_search_command",
]
