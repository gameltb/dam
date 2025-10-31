"""Semantic search-related functions for DAM."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
import torch
from dam.core.transaction import WorldTransaction
from dam.models.core.entity import Entity

# Import necessary tag components and functions to fetch tags
from dam_sire.resource import SireResource
from transformers import AutoModel, AutoTokenizer

from dam_semantic.models.text_embedding_component import (
    BaseSpecificEmbeddingComponent,
    ModelHyperparameters,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_PARAMS: ModelHyperparameters = {}


def mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling on the token embeddings.

    Args:
        model_output: The output from the transformer model.
        attention_mask: The attention mask for the input tokens.

    Returns:
        The pooled sentence embedding.

    """
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


async def generate_embedding(
    sire_resource: SireResource,
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    params: ModelHyperparameters | None = None,
) -> np.ndarray | None:
    """
    Generate an embedding for a given text using a transformer model.

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
        tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore
        model = AutoModel.from_pretrained(model_name, **(params or {}))  # type: ignore

        with sire_resource.auto_manage(model) as managed_model_wrapper:  # type: ignore
            managed_model = managed_model_wrapper.get_manage_object()  # type: ignore
            if managed_model is None:
                logger.error("Managed model is None, cannot encode text.")
                return None

            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")  # type: ignore

            device = next(iter(managed_model.parameters())).device  # type: ignore
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}  # type: ignore

            with torch.no_grad():
                model_output = managed_model(**encoded_input)  # type: ignore

            sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])  # type: ignore
            return sentence_embeddings.cpu().numpy().flatten()  # type: ignore

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
    sire_resource: SireResource,  # noqa: ARG001
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
    sire_resource: SireResource,  # noqa: ARG001
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
