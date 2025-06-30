import asyncio # Added asyncio
import logging
from typing import List, Dict, Any, Optional, Type, Tuple, TypedDict # Added TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dam.models.semantic import TextEmbeddingComponent
from dam.models.core.base_component import BaseComponent
from dam.models.core.entity import Entity # For type hinting if needed
from dam.services import ecs_service

logger = logging.getLogger(__name__)

# Initialize the model globally or within a class that manages it.
# Using a global variable for simplicity here, but for production, consider
# managing model loading more robustly (e.g., as a resource or singleton).
# Default model: 'all-MiniLM-L6-v2' (384 dimensions)
# Other options: 'multi-qa-MiniLM-L6-cos-v1' (good for semantic search)
DEFAULT_MODEL_NAME = 'all-MiniLM-L6-v2'
_model_cache: Dict[str, SentenceTransformer] = {}

def _load_model_sync(model_name: str) -> SentenceTransformer:
    """Synchronous helper to load the model, intended to be patched for tests."""
    logger.info(f"Attempting to load SentenceTransformer model: {model_name} (sync)")
    # This is where the actual SentenceTransformer class is instantiated.
    # The mock should replace SentenceTransformer before this function is called.
    model = SentenceTransformer(model_name)
    logger.info(f"Model {model_name} loaded successfully via sync helper.")
    return model

async def get_sentence_transformer_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Loads and caches a SentenceTransformer model.
    """
    if model_name not in _model_cache:
        try:
            # Run the synchronous model loading in a thread to avoid blocking asyncio event loop
            # if the actual loading (not mocked) takes time.
            # For tests, with MockSentenceTransformer, this should be quick.
            loop = asyncio.get_event_loop()
            # The key is that SentenceTransformer() is called within _load_model_sync,
            # which is what our test mock patches.
            _model_cache[model_name] = await loop.run_in_executor(None, _load_model_sync, model_name)
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}", exc_info=True)
            raise
    return _model_cache[model_name]

async def generate_embedding(text: str, model_name: str = DEFAULT_MODEL_NAME) -> Optional[np.ndarray]:
    """
    Generates a text embedding for the given text using the specified model.
    Returns a numpy array of floats, or None if generation fails.
    """
    if not text or not text.strip():
        logger.warning("Cannot generate embedding for empty or whitespace-only text.")
        return None
    try:
        model = await get_sentence_transformer_model(model_name)
        # Expect single text string here based on previous logic flow
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding for text '{text[:100]}...': {e}", exc_info=True)
        return None

def convert_embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Converts a numpy float32 embedding to bytes."""
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32) # Ensure float32 for consistent byte representation
    return embedding.tobytes()

def convert_bytes_to_embedding(embedding_bytes: bytes, dtype=np.float32) -> np.ndarray:
    """Converts bytes back to a numpy embedding."""
    return np.frombuffer(embedding_bytes, dtype=dtype)


class BatchTextItem(TypedDict):
    component_name: str
    field_name: str
    text_content: str


async def update_text_embeddings_for_entity(
    session: AsyncSession,
    entity_id: int,
    text_fields_map: Dict[str, Any], # e.g., {"ComponentName.field_name": "text content", ...}
    model_name: str = DEFAULT_MODEL_NAME,
    batch_texts: Optional[List[BatchTextItem]] = None,
) -> List[TextEmbeddingComponent]:
    """
    Generates and stores TextEmbeddingComponents for an entity based on a map of its text fields
    or a pre-compiled batch of texts.

    Args:
        session: The SQLAlchemy async session.
        entity_id: The ID of the entity to process.
        text_fields_map: A dictionary where keys are "ComponentName.FieldName" and values are the text content.
                         Used if `batch_texts` is not provided.
        model_name: The name of the sentence transformer model to use.
        batch_texts: Optional pre-compiled list of BatchTextItem dictionaries.
                     If provided, `text_fields_map` is ignored. This is useful for batch processing.

    Returns:
        A list of created or updated TextEmbeddingComponent instances.
    """
    processed_embeddings: List[TextEmbeddingComponent] = []

    current_batch_items: List[BatchTextItem] = []

    if batch_texts:
        current_batch_items = batch_texts
    else:
        for source_key, text_content_val in text_fields_map.items():
            if not text_content_val or not isinstance(text_content_val, str) or not text_content_val.strip():
                logger.debug(f"Skipping empty or invalid text content for {source_key} on entity {entity_id}")
                continue
            try:
                comp_name, field_name = source_key.split('.', 1)
                current_batch_items.append(
                    BatchTextItem(component_name=comp_name, field_name=field_name, text_content=text_content_val)
                )
            except ValueError:
                logger.warning(f"Invalid source_key format '{source_key}'. Should be 'ComponentName.FieldName'. Skipping.")
                continue

    if not current_batch_items:
        logger.info(f"No valid text fields to embed for entity {entity_id}.")
        return processed_embeddings

    # Batch encode all texts for the entity
    all_text_contents = [item["text_content"] for item in current_batch_items]
    embeddings_np_list: Optional[List[np.ndarray]] = None # Initialize as Optional List

    # Ensure model is loaded before attempting to encode
    try:
        # This time, we ensure the model is loaded, then pass all texts at once to encode
        # This is more efficient for sentence-transformers
        model = await get_sentence_transformer_model(model_name)
        # The model.encode method itself can handle a list of texts and return a list of embeddings
        embeddings_np_list = model.encode(all_text_contents, convert_to_numpy=True)

        if embeddings_np_list is None or len(embeddings_np_list) != len(all_text_contents):
             logger.error(f"Batch embedding generation returned unexpected result for entity {entity_id}. "
                          f"Expected {len(all_text_contents)} embeddings, got {len(embeddings_np_list) if embeddings_np_list is not None else 'None'}.")
             # Decide if to proceed with partial results or fail all. For now, fail all for this batch.
             return processed_embeddings

    except Exception as e: # Model loading or encoding failed
        logger.error(f"Cannot proceed with embedding generation for entity {entity_id} due to: {e}", exc_info=True)
        return processed_embeddings # Return empty list as no embeddings could be generated


    for i, batch_item in enumerate(current_batch_items):
        comp_name = batch_item["component_name"]
        field_name = batch_item["field_name"]
        # text_content = batch_item["text_content"] # Not needed here, but available

        embedding_np = embeddings_np_list[i] # Directly use the i-th embedding from the batch result

        # This check might be redundant if model.encode guarantees non-None for valid inputs,
        # but good for safety if any individual text in the batch could cause a specific failure
        # that results in a None or placeholder in the output list.
        # However, sentence-transformers usually throws an error or returns valid (possibly zero) vectors.
        # For simplicity, we assume valid ndarray outputs if no exception during model.encode(list).
        # If an individual text caused an issue, it's more likely to be an exception from model.encode
        # or a "bad" vector (e.g. zeros) rather than a None in the list.

        embedding_bytes = convert_embedding_to_bytes(embedding_np)

        # Check if an embedding for this source already exists and update it, or create a new one.
        # This simplistic check might need to be more robust (e.g., if multiple embeddings per source are allowed).
        existing_embeddings = await ecs_service.get_components_by_value(
            session,
            entity_id,
            TextEmbeddingComponent,
            attributes_values={ # Explicitly pass as attributes_values
                "model_name": model_name,
                "source_component_name": comp_name,
                "source_field_name": field_name,
            },
        )

        if existing_embeddings:
            # Update the first existing one found
            emb_comp = existing_embeddings[0]
            if emb_comp.embedding_vector != embedding_bytes:
                emb_comp.embedding_vector = embedding_bytes
                session.add(emb_comp)
                logger.info(f"Updated TextEmbeddingComponent for entity {entity_id}, source: {comp_name}.{field_name}, model: {model_name}")
                processed_embeddings.append(emb_comp)
            else:
                logger.debug(f"TextEmbeddingComponent for entity {entity_id}, source: {comp_name}.{field_name} is already up-to-date.")
                processed_embeddings.append(emb_comp) # Still include it as "processed"
        else:
            emb_comp = TextEmbeddingComponent(
                embedding_vector=embedding_bytes,
                model_name=model_name,
                source_component_name=comp_name,
                source_field_name=field_name,
            )
            await ecs_service.add_component_to_entity(session, entity_id, emb_comp)
            logger.info(f"Created TextEmbeddingComponent for entity {entity_id}, source: {comp_name}.{field_name}, model: {model_name}")
            processed_embeddings.append(emb_comp)

    # The session flush will be handled by the WorldScheduler or calling system.
    return processed_embeddings


async def get_text_embeddings_for_entity(
    session: AsyncSession,
    entity_id: int,
    model_name: Optional[str] = None
) -> List[TextEmbeddingComponent]:
    """
    Retrieves all TextEmbeddingComponents for a given entity, optionally filtered by model name.
    """
    if model_name:
        # Filter by model name
        return await ecs_service.get_components_by_value(
            session,
            entity_id,
            TextEmbeddingComponent,
            attributes_values={"model_name": model_name}
        )
    else:
        # Get all TextEmbeddingComponents for the entity
        return await ecs_service.get_components(
            session,
            entity_id,
            TextEmbeddingComponent
        )

async def find_similar_entities_by_text_embedding(
    session: AsyncSession,
    query_text: str,
    top_n: int = 10,
    model_name: str = DEFAULT_MODEL_NAME,
    # target_source_component: Optional[str] = None, # Future: filter by which field was embedded
    # target_source_field: Optional[str] = None,     # Future: filter by which field was embedded
) -> List[Tuple[Entity, float, TextEmbeddingComponent]]: # (Entity, similarity_score, TextEmbeddingComponent)
    """
    Finds entities similar to the query text using cosine similarity on stored text embeddings.
    This is a brute-force search and may be slow on large datasets.
    """
    query_embedding_np = await generate_embedding(query_text, model_name)
    if query_embedding_np is None:
        logger.error(f"Could not generate embedding for query text: '{query_text[:100]}...'")
        return []

    # Fetch all relevant TextEmbeddingComponents
    # This could be very large. Consider pagination or more targeted fetching if needed.
    all_embedding_components_stmt = select(TextEmbeddingComponent).where(
        TextEmbeddingComponent.model_name == model_name
    )
    # TODO: Add filters for target_source_component/field if provided

    result = await session.execute(all_embedding_components_stmt)
    all_embedding_components = result.scalars().all()

    if not all_embedding_components:
        logger.info(f"No text embeddings found for model {model_name} to compare against.")
        return []

    similarities = []
    for emb_comp in all_embedding_components:
        db_embedding_np = convert_bytes_to_embedding(emb_comp.embedding_vector)

        # Cosine similarity: (A dot B) / (||A|| * ||B||)
        # sentence-transformers.util.cos_sim can also be used if embeddings are torch tensors or numpy arrays
        # from sentence_transformers.util import cos_sim
        # score = cos_sim(query_embedding_np, db_embedding_np).item() # .item() if it's a single value tensor/array

        dot_product = np.dot(query_embedding_np, db_embedding_np)
        norm_query = np.linalg.norm(query_embedding_np)
        norm_db = np.linalg.norm(db_embedding_np)

        if norm_query == 0 or norm_db == 0: # Avoid division by zero
            score = 0.0
        else:
            score = dot_product / (norm_query * norm_db)

        # Filter out very low scores if needed, though sorting will handle it.
        # if score > some_threshold:
        similarities.append((emb_comp.entity_id, score, emb_comp))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get top N results and fetch their entities
    top_results: List[Tuple[Entity, float, TextEmbeddingComponent]] = []
    entity_ids_processed = set()

    for entity_id, score, emb_comp_instance in similarities:
        if len(top_results) >= top_n:
            break
        if entity_id in entity_ids_processed: # Avoid duplicate entities if multiple embeddings from same entity match
            continue

        entity = await ecs_service.get_entity(session, entity_id)
        if entity:
            top_results.append((entity, float(score), emb_comp_instance))
            entity_ids_processed.add(entity_id)

    return top_results
