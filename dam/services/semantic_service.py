import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dam.core import get_default_world  # To get ModelExecutionManager resource
from dam.core.model_manager import ModelExecutionManager
from dam.models.core.entity import Entity
from dam.models.semantic import (
    EMBEDDING_MODEL_REGISTRY,  # For default params
    BaseSpecificEmbeddingComponent,
    ModelHyperparameters,
    get_embedding_component_class,
)

# Import necessary tag components and service to fetch tags
from dam.models.tags import ModelGeneratedTagLinkComponent, TagConceptComponent
from dam.services import ecs_service
from dam.services.tag_service import get_tags_for_entity  # For manual tags

# For model-generated tags, we might need a function in tagging_service or query directly for now
# from dam.services.tagging_service import get_model_generated_tags_for_entity # Assuming this will exist

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"  # This should match a key in EMBEDDING_MODEL_REGISTRY
DEFAULT_MODEL_PARAMS: ModelHyperparameters = {}  # Default params, if any, might be better fetched from registry

SENTENCE_TRANSFORMER_IDENTIFIER = "sentence_transformer"


def _load_sentence_transformer_model_sync(
    model_name_or_path: str, params: Optional[Dict[str, Any]] = None
) -> SentenceTransformer:
    """
    Synchronous helper to load a SentenceTransformer model.
    `params` can include device, cache_folder, etc. for SentenceTransformer constructor.
    """
    logger.info(f"Attempting to load SentenceTransformer model: {model_name_or_path} with params {params}")
    # If device is not in params, ModelExecutionManager might set a default one later,
    # or SentenceTransformer might pick one.
    # For now, pass all params.
    device = params.pop("device", None)  # Pop device if present, ST handles it.
    model = SentenceTransformer(model_name_or_path, device=device, **(params or {}))
    logger.info(f"SentenceTransformer model {model_name_or_path} loaded successfully.")
    return model


async def get_sentence_transformer_model(
    model_execution_manager: ModelExecutionManager, # Added: MEM must be passed in
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    params: Optional[ModelHyperparameters] = None,  # Conceptual params
    # world_name: Optional[str] = None, # Removed: MEM is global, world_name not needed for its context
) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model using the provided ModelExecutionManager.
    `params` are conceptual hyperparameters; they might be mapped to loader params.
    """
    # Ensure loader is registered if not already
    # This registration is now idempotent or should be handled at MEM initialization.
    if SENTENCE_TRANSFORMER_IDENTIFIER not in model_execution_manager._model_loaders:
        model_execution_manager.register_model_loader(
            SENTENCE_TRANSFORMER_IDENTIFIER, _load_sentence_transformer_model_sync
        )

    # Map conceptual params to loader params if needed. For ST, model_name_or_path is key.
    # `params` might include things like `device` or `cache_folder` for the loader.
    loader_params = params.copy() if params else {}
    if "device" not in loader_params:  # Add default device preference if not specified
        loader_params["device"] = model_execution_manager.get_model_device_preference()

    return await model_execution_manager.get_model(
        model_identifier=SENTENCE_TRANSFORMER_IDENTIFIER, model_name_or_path=model_name_or_path, params=loader_params
    )


async def generate_embedding(
    model_execution_manager: ModelExecutionManager, # Added
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    params: Optional[ModelHyperparameters] = None,
    # world_name: Optional[str] = None, # Removed
) -> Optional[np.ndarray]:
    """
    Generates a text embedding for the given text using the specified model and parameters,
    via the provided ModelExecutionManager.
    Returns a numpy array of floats, or None if generation fails.
    """
    if not text or not text.strip():
        logger.warning("Cannot generate embedding for empty or whitespace-only text.")
        return None
    try:
        model = await get_sentence_transformer_model(
            model_execution_manager, model_name, params # Pass MEM
        )
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
        logger.error(
            f"Error generating embedding for text '{text[:100]}...' with model {model_name}, params {params}: {e}",
            exc_info=True,
        )
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
    field_name: str  # Source field name
    text_content: str


async def update_text_embeddings_for_entity(
    session: AsyncSession,
    entity_id: int,
    text_fields_map: Dict[str, Any],  # e.g., {"ComponentName.field_name": "text content"}
    model_execution_manager: ModelExecutionManager, # Moved up
    model_name: str = DEFAULT_MODEL_NAME,
    model_params: Optional[ModelHyperparameters] = None,  # Conceptual params for model version
    batch_texts: Optional[List[BatchTextItem]] = None,
    # Configuration for including tags
    include_manual_tags: bool = True,
    include_model_tags_config: Optional[
        List[Dict[str, Any]]
    ] = None,  # e.g., [{"model_name": "wd-v1...", "min_confidence": 0.5}]
    tag_concatenation_strategy: str = " [TAGS] {tags_string}",  # How to append tags
    # world_name: Optional[str] = None, # Removed
) -> List[BaseSpecificEmbeddingComponent]:
    """
    Generates and stores specific TextEmbeddingComponents for an entity using the provided ModelExecutionManager.
    Optionally fetches manual and/or model-generated tags associated with the entity,
    concatenates them to the text fields, and then generates embeddings.
    Uses the registry to determine the correct table/component class for storing the embedding.
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
        else:  # Should not happen if get_embedding_component_class succeeded
            model_params = {}

    processed_embeddings: List[BaseSpecificEmbeddingComponent] = []
    current_batch_items: List[BatchTextItem] = []
    augmented_source_field_suffix = ""

    # Prepare tag string if needed
    tags_to_concatenate_str = ""
    if include_manual_tags or include_model_tags_config:
        all_tag_names_for_entity: List[str] = []

        # Fetch manual tags
        if include_manual_tags:
            manual_tag_tuples = await get_tags_for_entity(
                session, entity_id
            )  # Returns List[Tuple[Entity, Optional[str]]]
            for tag_entity, _ in manual_tag_tuples:
                tag_concept = await ecs_service.get_component(session, tag_entity.id, TagConceptComponent)
                if tag_concept and tag_concept.tag_name:
                    all_tag_names_for_entity.append(tag_concept.tag_name)
            if manual_tag_tuples:
                augmented_source_field_suffix += "_manualtags"

        # Fetch model-generated tags
        if include_model_tags_config:
            for model_tag_conf in include_model_tags_config:
                model_tag_name = model_tag_conf.get("model_name")
                min_confidence = model_tag_conf.get("min_confidence", 0.0)
                if not model_tag_name:
                    continue

                # Query for ModelGeneratedTagLinkComponent
                stmt = select(ModelGeneratedTagLinkComponent).where(
                    ModelGeneratedTagLinkComponent.entity_id == entity_id,
                    ModelGeneratedTagLinkComponent.source_model_name == model_tag_name,
                )
                if min_confidence > 0:
                    stmt = stmt.where(ModelGeneratedTagLinkComponent.confidence >= min_confidence)

                result = await session.execute(stmt)
                model_tag_links = result.scalars().all()

                model_tags_fetched_this_model = []
                for link in model_tag_links:
                    tag_concept = await ecs_service.get_component(session, link.tag_concept_id, TagConceptComponent)
                    if tag_concept and tag_concept.tag_name:
                        model_tags_fetched_this_model.append(tag_concept.tag_name)

                if model_tags_fetched_this_model:
                    all_tag_names_for_entity.extend(model_tags_fetched_this_model)
                    # Sanitize model_tag_name for suffix
                    safe_model_suffix = "".join(c if c.isalnum() else "_" for c in model_tag_name)
                    augmented_source_field_suffix += f"_model_{safe_model_suffix}"

        if all_tag_names_for_entity:
            # Remove duplicates while preserving order (Python 3.7+)
            unique_tags = list(dict.fromkeys(all_tag_names_for_entity))
            tags_to_concatenate_str = ", ".join(unique_tags)

    # Prepare batch items
    if batch_texts:  # If batch_texts is provided, assume it's already prepared (possibly with tags)
        current_batch_items = batch_texts
        # If using batch_texts, augmented_source_field_suffix might not be accurate unless batch_texts also reflect it.
        # For simplicity, if batch_texts is used, we assume tag augmentation is handled by the caller or not desired for this specific call.
        augmented_source_field_suffix = "_batch"  # Indicate it's from a pre-compiled batch
    else:
        for source_key, text_content_val in text_fields_map.items():
            if not text_content_val or not isinstance(text_content_val, str) or not text_content_val.strip():
                logger.debug(f"Skipping empty/invalid text for {source_key} on entity {entity_id}")
                continue

            final_text_to_embed = text_content_val
            if tags_to_concatenate_str:
                # Apply the concatenation strategy
                final_text_to_embed += tag_concatenation_strategy.format(tags_string=tags_to_concatenate_str)

            try:
                comp_name, field_name = source_key.split(".", 1)
                current_batch_items.append(
                    BatchTextItem(component_name=comp_name, field_name=field_name, text_content=final_text_to_embed)
                )
            except ValueError:
                logger.warning(f"Invalid source_key '{source_key}'. Skipping.")
                continue

    if not current_batch_items:
        logger.info(
            f"No valid text fields (with or without tags) to embed for entity {entity_id} with model {model_name}."
        )
        return processed_embeddings

    # Generate embeddings for all prepared text contents
    all_text_contents_for_model = [item["text_content"] for item in current_batch_items]
    embeddings_np_list: Optional[List[np.ndarray]] = None
    try:
        model_instance = await get_sentence_transformer_model(
            model_execution_manager, model_name, model_params # Pass MEM
        )

        # Get optimal batch size from model_manager if possible (conceptual)
        # For now, SentenceTransformer handles its own batching internally for encode()
        # If we were to implement manual batching:
        # world = get_default_world(world_name)
        # model_manager = world.get_resource(ModelExecutionManager)
        # item_size_estimate = 5 # MB, very rough estimate for a text string tensor
        # batch_size = model_manager.get_optimal_batch_size(SENTENCE_TRANSFORMER_IDENTIFIER, model_name, item_size_estimate)
        # # And then loop through all_text_contents_for_model in batches of batch_size

        embeddings_np_list = model_instance.encode(all_text_contents_for_model, convert_to_numpy=True)
        if embeddings_np_list is None or len(embeddings_np_list) != len(all_text_contents_for_model):
            logger.error(f"Batch embedding generation returned unexpected result for entity {entity_id}.")
            return processed_embeddings
    except Exception as e:
        logger.error(f"Embedding generation failed for entity {entity_id}, model {model_name}: {e}", exc_info=True)
        return processed_embeddings

    # Store each generated embedding
    for i, batch_item in enumerate(current_batch_items):
        original_comp_name = batch_item["component_name"]
        original_field_name = batch_item["field_name"]

        # Adjust field_name if tags were added and not using pre-compiled batch_texts
        # For pre-compiled batch_texts, the source field name is taken as is.
        final_field_name = original_field_name
        if (
            not batch_texts and tags_to_concatenate_str
        ):  # only add suffix if tags were actually added and not from batch
            final_field_name += augmented_source_field_suffix
        elif batch_texts and augmented_source_field_suffix == "_batch":  # if it IS from batch, use original field name
            final_field_name = original_field_name

        embedding_np = embeddings_np_list[i]
        embedding_bytes = convert_embedding_to_bytes(embedding_np)

        existing_embeddings = await ecs_service.get_components_by_value(
            session,
            entity_id,
            EmbeddingComponentClass,
            attributes_values={
                "source_component_name": original_comp_name,
                "source_field_name": final_field_name,  # Use potentially augmented field name
            },
        )

        if existing_embeddings:
            emb_comp = existing_embeddings[0]
            if emb_comp.embedding_vector != embedding_bytes:
                emb_comp.embedding_vector = embedding_bytes
                session.add(emb_comp)
                logger.info(
                    f"Updated {EmbeddingComponentClass.__name__} for entity {entity_id}, src: {original_comp_name}.{final_field_name}"
                )
            else:
                logger.debug(
                    f"{EmbeddingComponentClass.__name__} for entity {entity_id}, src: {original_comp_name}.{final_field_name} is up-to-date."
                )
            processed_embeddings.append(emb_comp)
        else:
            emb_comp = EmbeddingComponentClass(
                embedding_vector=embedding_bytes,
                source_component_name=original_comp_name,
                source_field_name=final_field_name,  # Use potentially augmented field name
            )
            await ecs_service.add_component_to_entity(session, entity_id, emb_comp)
            logger.info(
                f"Created {EmbeddingComponentClass.__name__} for entity {entity_id}, src: {original_comp_name}.{final_field_name}"
            )
            processed_embeddings.append(emb_comp)

    return processed_embeddings


async def get_text_embeddings_for_entity(
    session: AsyncSession,
    entity_id: int,
    model_name: str,  # Must specify which table to query
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
    model_execution_manager: ModelExecutionManager, # Added
    model_name: str,  # Must specify which model/table to search
    model_params: Optional[ModelHyperparameters] = None,
    top_n: int = 10,
    # world_name: Optional[str] = None, # Removed
) -> List[Tuple[Entity, float, BaseSpecificEmbeddingComponent]]:
    """
    Finds entities similar to the query text using cosine similarity on stored text embeddings,
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

    query_embedding_np = await generate_embedding(
        model_execution_manager, query_text, model_name, model_params # Pass MEM
    )
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
