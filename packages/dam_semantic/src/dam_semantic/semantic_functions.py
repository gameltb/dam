"""Semantic search-related functions for DAM."""

import logging
from typing import Any, TypedDict

import numpy as np
from dam.core.transaction import WorldTransaction
from dam.models.core.entity import Entity

# Import necessary tag components and functions to fetch tags
from dam_sire.resource import SireResource
from sentence_transformers import SentenceTransformer

from dam_semantic.models.text_embedding_component import (
    BaseSpecificEmbeddingComponent,
    ModelHyperparameters,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_PARAMS: ModelHyperparameters = {}

SENTENCE_TRANSFORMER_IDENTIFIER = "sentence_transformer"


async def generate_embedding(
    sire_resource: "SireResource",
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    params: ModelHyperparameters | None = None,
) -> np.ndarray | None:
    """
    Generate an embedding for a given text using a sentence transformer model.

    Args:
        sire_resource: The Sire resource for managing models.
        text: The text to embed.
        model_name: The name of the sentence transformer model to use.
        params: Additional parameters for the model.

    Returns:
        The generated embedding as a numpy array, or None if an error occurred.

    """
    if not text or not text.strip():
        return None
    try:
        if params:
            params.pop("dimensions", None)
        model = SentenceTransformer(model_name, **(params or {}))
        with sire_resource.auto_manage(model) as managed_model_wrapper:
            managed_model = managed_model_wrapper.get_manage_object()
            if managed_model is None:
                logger.error("Managed model is None, cannot encode text.")
                return None
            return managed_model.encode(text, convert_to_numpy=True)
    except Exception:
        logger.exception("Error generating embedding")
        return None


def convert_embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """
    Convert a numpy array embedding to bytes.

    Args:
        embedding: The numpy array to convert.

    Returns:
        The embedding as bytes.

    """
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return embedding.tobytes()


def convert_bytes_to_embedding(embedding_bytes: bytes, dtype: Any = np.float32) -> np.ndarray:
    """
    Convert bytes to a numpy array embedding.

    Args:
        embedding_bytes: The bytes to convert.
        dtype: The numpy data type of the embedding.

    Returns:
        The embedding as a numpy array.

    """
    return np.frombuffer(embedding_bytes, dtype=dtype)


class BatchTextItem(TypedDict):
    """Represents a single item for batch text processing."""

    component_name: str
    field_name: str
    text_content: str


async def update_text_embeddings_for_entity(
    transaction: WorldTransaction,  # noqa: ARG001
    entity_id: int,  # noqa: ARG001
    text_fields_map: dict[str, Any],  # noqa: ARG001
    sire_resource: "SireResource",  # noqa: ARG001
    model_name: str = DEFAULT_MODEL_NAME,  # noqa: ARG001
    model_params: ModelHyperparameters | None = None,  # noqa: ARG001
    batch_texts: list[BatchTextItem] | None = None,  # noqa: ARG001
    include_manual_tags: bool = True,  # noqa: ARG001
    include_model_tags_config: list[dict[str, Any]] | None = None,  # noqa: ARG001
    tag_concatenation_strategy: str = " [TAGS] {tags_string}",  # noqa: ARG001
) -> list[BaseSpecificEmbeddingComponent]:
    """
    Update text embeddings for a given entity.

    Note: This function is not fully implemented yet.
    """
    logger.warning("update_text_embeddings_for_entity is not fully implemented with sire yet.")
    return []


async def find_similar_entities_by_text_embedding(
    transaction: WorldTransaction,  # noqa: ARG001
    query_text: str,  # noqa: ARG001
    sire_resource: "SireResource",  # noqa: ARG001
    model_name: str,  # noqa: ARG001
    model_params: ModelHyperparameters | None = None,  # noqa: ARG001
    top_n: int = 10,  # noqa: ARG001
) -> list[tuple[Entity, float, BaseSpecificEmbeddingComponent]]:
    """
    Find similar entities by text embedding.

    Note: This function is not fully implemented yet.
    """
    logger.warning("find_similar_entities_by_text_embedding is not fully implemented with sire yet.")
    return []
