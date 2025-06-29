import logging
from typing import List, Dict, Any, Optional, Type

import numpy as np
from sentence_transformers import SentenceTransformer
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

def get_sentence_transformer_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Loads and caches a SentenceTransformer model.
    """
    if model_name not in _model_cache:
        try:
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            _model_cache[model_name] = SentenceTransformer(model_name)
            logger.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}", exc_info=True)
            raise
    return _model_cache[model_name]

def generate_embedding(text: str, model_name: str = DEFAULT_MODEL_NAME) -> Optional[np.ndarray]:
    """
    Generates a text embedding for the given text using the specified model.
    Returns a numpy array of floats, or None if generation fails.
    """
    if not text or not text.strip():
        logger.warning("Cannot generate embedding for empty or whitespace-only text.")
        return None
    try:
        model = get_sentence_transformer_model(model_name)
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


async def update_text_embeddings_for_entity(
    session: AsyncSession,
    entity_id: int,
    text_fields_map: Dict[str, Any], # e.g., {"ComponentName.field_name": "text content", ...}
    model_name: str = DEFAULT_MODEL_NAME,
    batch_texts: Optional[List[Tuple[str, str, str]]] = None, # (component_name, field_name, text_content)
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
        batch_texts: Optional pre-compiled list of (component_name, field_name, text_content) tuples.
                     If provided, `text_fields_map` is ignored. This is useful for batch processing.

    Returns:
        A list of created or updated TextEmbeddingComponent instances.
    """
    processed_embeddings: List[TextEmbeddingComponent] = []

    texts_to_embed: List[Tuple[str, str, str]] = [] # (component_name, field_name, text_content)

    if batch_texts:
        texts_to_embed = batch_texts
    else:
        for source_key, text_content in text_fields_map.items():
            if not text_content or not isinstance(text_content, str) or not text_content.strip():
                logger.debug(f"Skipping empty or invalid text content for {source_key} on entity {entity_id}")
                continue
            try:
                comp_name, field_name = source_key.split('.', 1)
                texts_to_embed.append((comp_name, field_name, text_content))
            except ValueError:
                logger.warning(f"Invalid source_key format '{source_key}'. Should be 'ComponentName.FieldName'. Skipping.")
                continue

    if not texts_to_embed:
        logger.info(f"No valid text fields to embed for entity {entity_id}.")
        return processed_embeddings

    # Batch encode all texts for the entity
    all_text_contents = [item[2] for item in texts_to_embed]

    # Ensure model is loaded before attempting to encode
    try:
        get_sentence_transformer_model(model_name)
    except Exception: # Model loading failed
        logger.error(f"Cannot proceed with embedding generation for entity {entity_id} due to model loading failure.")
        return processed_embeddings # Return empty list as no embeddings could be generated

    embeddings_np = generate_embedding(all_text_contents, model_name=model_name) # generate_embedding handles list input

    if embeddings_np is None or len(embeddings_np) != len(all_text_contents):
        logger.error(f"Batch embedding generation failed or returned unexpected number of embeddings for entity {entity_id}.")
        return processed_embeddings

    for i, (comp_name, field_name, text_content) in enumerate(texts_to_embed):
        embedding_np = embeddings_np[i]
        embedding_bytes = convert_embedding_to_bytes(embedding_np)

        # Check if an embedding for this source already exists and update it, or create a new one.
        # This simplistic check might need to be more robust (e.g., if multiple embeddings per source are allowed).
        existing_embeddings = await ecs_service.find_components_by_attributes(
            session,
            entity_id,
            TextEmbeddingComponent,
            {
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
    filters = {}
    if model_name:
        filters["model_name"] = model_name

    return await ecs_service.find_components_by_attributes(
        session,
        entity_id,
        TextEmbeddingComponent,
        filters if filters else None # Pass None if no filters to get all
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
    query_embedding_np = generate_embedding(query_text, model_name)
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
