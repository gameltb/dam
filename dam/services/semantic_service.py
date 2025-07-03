import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Type, FrozenSet

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import class_mapper # Used for inspecting mapped properties

from dam.models.core.entity import Entity
from dam.models.semantic import (
    get_embedding_component_class,
    BaseSpecificEmbeddingComponent,
    ModelHyperparameters,
    EMBEDDING_MODEL_REGISTRY, # For default params
)
from dam.services import ecs_service

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2" # This should match a key in EMBEDDING_MODEL_REGISTRY
DEFAULT_MODEL_PARAMS: ModelHyperparameters = {} # Default params, if any, might be better fetched from registry


# Cache key: (model_name, frozenset(params.items()))
_model_cache: Dict[Tuple[str, FrozenSet[Tuple[str, Any]]], SentenceTransformer] = {}


def _load_model_sync(model_name: str, model_load_params: Optional[Dict[str, Any]] = None) -> SentenceTransformer:
    """
    Synchronous helper to load the model.
    `model_load_params` are parameters passed directly to SentenceTransformer constructor if any.
    """
    logger.info(f"Attempting to load SentenceTransformer model: {model_name} with params {model_load_params} (sync)")
    # For now, we assume model_name is sufficient for SentenceTransformer,
    # and model_load_params might be for future use (e.g. device, cache_folder)
    # If specific params from ModelHyperparameters directly map to ST constructor args, pass them.
    model = SentenceTransformer(model_name, **(model_load_params or {}))
    logger.info(f"Model {model_name} loaded successfully via sync helper.")
    return model


async def get_sentence_transformer_model(
    model_name: str = DEFAULT_MODEL_NAME,
    params: Optional[ModelHyperparameters] = None,
) -> SentenceTransformer:
    """
    Loads and caches a SentenceTransformer model based on its name and relevant parameters.
    `params` here are the conceptual hyperparameters of the embedding (e.g., dimension),
    not necessarily direct arguments to SentenceTransformer constructor unless they overlap.
    """
    # Use default params from registry if not provided, ensuring consistency
    if params is None:
        registry_entry = EMBEDDING_MODEL_REGISTRY.get(model_name)
        if registry_entry:
            params = registry_entry.get("default_params", {})
        else:
            # This case should ideally not happen if model_name is always valid and registered
            logger.warning(f"Model {model_name} not found in registry, using empty params for caching.")
            params = {}

    # Some parameters might be relevant for model loading itself (e.g. device, specific sub-model path)
    # For now, we assume model_name is the primary identifier for SentenceTransformer library.
    # The `params` dict helps differentiate between variations if needed for caching or logic,
    # but might not all be passed to `SentenceTransformer()`.
    model_load_params = {} # Extract relevant params for SentenceTransformer() if any from `params`

    cache_key = (model_name, frozenset(params.items()))

    if cache_key not in _model_cache:
        try:
            loop = asyncio.get_event_loop()
            _model_cache[cache_key] = await loop.run_in_executor(
                None, _load_model_sync, model_name, model_load_params
            )
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name} with params {params}: {e}", exc_info=True)
            raise
    return _model_cache[cache_key]


async def generate_embedding(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    params: Optional[ModelHyperparameters] = None,
) -> Optional[np.ndarray]:
    """
    Generates a text embedding for the given text using the specified model and parameters.
    Returns a numpy array of floats, or None if generation fails.
    """
    if not text or not text.strip():
        logger.warning("Cannot generate embedding for empty or whitespace-only text.")
        return None
    try:
        model = await get_sentence_transformer_model(model_name, params)
        embedding = model.encode(text, convert_to_numpy=True)
        # TODO: Validate embedding dimension against expected from params if applicable
        # registry_entry = EMBEDDING_MODEL_REGISTRY.get(model_name)
        # if registry_entry and params:
        #     expected_dim = params.get("dimensions") # or registry_entry["default_params"].get("dimensions")
        #     if expected_dim and embedding.shape[0] != expected_dim:
        #         logger.error(f"Embedding for {model_name} has unexpected dimension {embedding.shape[0]}, expected {expected_dim}")
        #         return None
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding for text '{text[:100]}...' with model {model_name}, params {params}: {e}", exc_info=True)
        return None


def convert_embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Converts a numpy float32 embedding to bytes."""
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return embedding.tobytes()


def convert_bytes_to_embedding(embedding_bytes: bytes, dtype=np.float32) -> np.ndarray:
    """Converts bytes back to a numpy embedding."""
    return np.frombuffer(embedding_bytes, dtype=dtype)


class BatchTextItem(TypedDict):
    component_name: str  # Source component name
    field_name: str      # Source field name
    text_content: str


async def update_text_embeddings_for_entity(
    session: AsyncSession,
    entity_id: int,
    text_fields_map: Dict[str, Any], # e.g., {"ComponentName.field_name": "text content"}
    model_name: str = DEFAULT_MODEL_NAME,
    model_params: Optional[ModelHyperparameters] = None, # Conceptual params for model version
    batch_texts: Optional[List[BatchTextItem]] = None,
) -> List[BaseSpecificEmbeddingComponent]:
    """
    Generates and stores specific TextEmbeddingComponents for an entity.
    Uses the registry to determine the correct table/component class.
    """
    # Determine the specific embedding component class
    EmbeddingComponentClass = get_embedding_component_class(model_name, model_params)
    if not EmbeddingComponentClass:
        logger.error(
            f"No embedding component class found for model '{model_name}' and params {model_params}. Skipping embedding update."
        )
        return []

    # Use default params from registry if not provided, for consistency in model interaction
    if model_params is None:
        registry_entry = EMBEDDING_MODEL_REGISTRY.get(model_name)
        if registry_entry:
            model_params = registry_entry.get("default_params", {})
        else: # Should not happen if get_embedding_component_class succeeded
            model_params = {}


    processed_embeddings: List[BaseSpecificEmbeddingComponent] = []
    current_batch_items: List[BatchTextItem] = []

    if batch_texts:
        current_batch_items = batch_texts
    else:
        for source_key, text_content_val in text_fields_map.items():
            if not text_content_val or not isinstance(text_content_val, str) or not text_content_val.strip():
                logger.debug(f"Skipping empty/invalid text for {source_key} on entity {entity_id}")
                continue
            try:
                comp_name, field_name = source_key.split(".", 1)
                current_batch_items.append(
                    BatchTextItem(component_name=comp_name, field_name=field_name, text_content=text_content_val)
                )
            except ValueError:
                logger.warning(f"Invalid source_key '{source_key}'. Skipping.")
                continue

    if not current_batch_items:
        logger.info(f"No valid text fields to embed for entity {entity_id} with model {model_name}.")
        return processed_embeddings

    all_text_contents = [item["text_content"] for item in current_batch_items]
    embeddings_np_list: Optional[List[np.ndarray]] = None

    try:
        # Pass model_params to get_sentence_transformer_model for consistent model instance usage
        model_instance = await get_sentence_transformer_model(model_name, model_params)
        embeddings_np_list = model_instance.encode(all_text_contents, convert_to_numpy=True)

        if embeddings_np_list is None or len(embeddings_np_list) != len(all_text_contents):
            logger.error(f"Batch embedding failed for entity {entity_id}, model {model_name}.")
            return processed_embeddings
    except Exception as e:
        logger.error(f"Embedding generation failed for entity {entity_id}, model {model_name}: {e}", exc_info=True)
        return processed_embeddings

    for i, batch_item in enumerate(current_batch_items):
        comp_name = batch_item["component_name"]
        field_name = batch_item["field_name"]
        embedding_np = embeddings_np_list[i]
        embedding_bytes = convert_embedding_to_bytes(embedding_np)

        # Query for existing component of the specific type
        # Since model_name and params define the table, they are not attributes for filtering within the table.
        existing_embeddings = await ecs_service.get_components_by_value(
            session,
            entity_id,
            EmbeddingComponentClass, # Query the specific table
            attributes_values={
                "source_component_name": comp_name,
                "source_field_name": field_name,
            },
        )

        if existing_embeddings:
            emb_comp = existing_embeddings[0]
            if emb_comp.embedding_vector != embedding_bytes:
                emb_comp.embedding_vector = embedding_bytes
                session.add(emb_comp)
                logger.info(
                    f"Updated {EmbeddingComponentClass.__name__} for entity {entity_id}, src: {comp_name}.{field_name}"
                )
            else:
                logger.debug(
                    f"{EmbeddingComponentClass.__name__} for entity {entity_id}, src: {comp_name}.{field_name} is up-to-date."
                )
            processed_embeddings.append(emb_comp)
        else:
            # Create new instance of the specific component class
            emb_comp = EmbeddingComponentClass(
                embedding_vector=embedding_bytes,
                source_component_name=comp_name,
                source_field_name=field_name,
            )
            await ecs_service.add_component_to_entity(session, entity_id, emb_comp)
            logger.info(
                f"Created {EmbeddingComponentClass.__name__} for entity {entity_id}, src: {comp_name}.{field_name}"
            )
            processed_embeddings.append(emb_comp)

    return processed_embeddings


async def get_text_embeddings_for_entity(
    session: AsyncSession,
    entity_id: int,
    model_name: str, # Must specify which table to query
    model_params: Optional[ModelHyperparameters] = None,
) -> List[BaseSpecificEmbeddingComponent]:
    """
    Retrieves specific TextEmbeddingComponents for an entity, from the table
    corresponding to model_name and model_params.
    """
    EmbeddingComponentClass = get_embedding_component_class(model_name, model_params)
    if not EmbeddingComponentClass:
        logger.warning(f"No component class for model {model_name}, params {model_params}. Cannot get embeddings.")
        return []

    # No need to filter by model_name/params in query, as table itself defines this.
    return await ecs_service.get_components(session, entity_id, EmbeddingComponentClass)


async def find_similar_entities_by_text_embedding(
    session: AsyncSession,
    query_text: str,
    model_name: str, # Must specify which model/table to search
    model_params: Optional[ModelHyperparameters] = None,
    top_n: int = 10,
) -> List[Tuple[Entity, float, BaseSpecificEmbeddingComponent]]:
    """
    Finds entities similar to the query text using cosine similarity on stored text embeddings
    from the table specified by model_name and model_params.
    """
    EmbeddingComponentClass = get_embedding_component_class(model_name, model_params)
    if not EmbeddingComponentClass:
        logger.error(f"No component class for model {model_name}, params {model_params}. Cannot find similar entities.")
        return []

    # Use default params from registry if not provided for consistency
    if model_params is None:
        registry_entry = EMBEDDING_MODEL_REGISTRY.get(model_name)
        if registry_entry:
            model_params = registry_entry.get("default_params", {})
        else:
            model_params = {}


    query_embedding_np = await generate_embedding(query_text, model_name, model_params)
    if query_embedding_np is None:
        logger.error(f"Could not generate query embedding for '{query_text[:100]}...' with model {model_name}.")
        return []

    # Fetch all embeddings from the specific table
    all_embedding_components_stmt = select(EmbeddingComponentClass)
    result = await session.execute(all_embedding_components_stmt)
    all_embedding_components = result.scalars().all()

    if not all_embedding_components:
        logger.info(f"No embeddings found in table for {EmbeddingComponentClass.__name__} to compare against.")
        return []

    similarities = []
    for emb_comp in all_embedding_components:
        db_embedding_np = convert_bytes_to_embedding(emb_comp.embedding_vector)
        dot_product = np.dot(query_embedding_np, db_embedding_np)
        norm_query = np.linalg.norm(query_embedding_np)
        norm_db = np.linalg.norm(db_embedding_np)

        if norm_query == 0 or norm_db == 0:
            score = 0.0
        else:
            score = dot_product / (norm_query * norm_db)
        similarities.append((emb_comp.entity_id, score, emb_comp))

    similarities.sort(key=lambda x: x[1], reverse=True)

    top_results: List[Tuple[Entity, float, BaseSpecificEmbeddingComponent]] = []
    entity_ids_processed = set()
    for entity_id, score, emb_comp_instance in similarities:
        if len(top_results) >= top_n:
            break
        if entity_id in entity_ids_processed:
            continue

        entity = await ecs_service.get_entity(session, entity_id)
        if entity:
            top_results.append((entity, float(score), emb_comp_instance))
            entity_ids_processed.add(entity_id)

    return top_results

# Cleanup: remove old TextEmbeddingComponent if it's truly replaced.
# from dam.models.semantic import TextEmbeddingComponent # This would be the old one
# Ensure all functions now use BaseSpecificEmbeddingComponent or its subclasses.
# The return types and type hints have been updated.
# The ecs_service calls now use the specific EmbeddingComponentClass.
# The old TextEmbeddingComponent is no longer referenced directly in this service.
# It might still be in dam.models.semantic.__init__ as OldTextEmbeddingComponent for now.
