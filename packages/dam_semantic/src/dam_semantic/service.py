import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from dam.models.core.entity import Entity

# Import necessary tag components and service to fetch tags
from dam.models.tags import ModelGeneratedTagLinkComponent, TagConceptComponent
from dam.services import ecs_service
from dam.services.tag_service import get_tags_for_entity  # For manual tags
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    EMBEDDING_MODEL_REGISTRY,
    BaseSpecificEmbeddingComponent,
    ModelHyperparameters,
    get_embedding_component_class,
)

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


def convert_embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Converts a numpy float32 embedding to bytes."""
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return embedding.tobytes()


def convert_bytes_to_embedding(embedding_bytes: bytes, dtype=np.float32) -> np.ndarray:
    """Converts bytes back to a numpy embedding."""
    return np.frombuffer(embedding_bytes, dtype=dtype)


class SemanticService:
    def __init__(self):
        self.DEFAULT_MODEL_NAME = DEFAULT_MODEL_NAME
        self.DEFAULT_MODEL_PARAMS = DEFAULT_MODEL_PARAMS
        self.SENTENCE_TRANSFORMER_IDENTIFIER = SENTENCE_TRANSFORMER_IDENTIFIER
        self._load_sentence_transformer_model_sync = _load_sentence_transformer_model_sync
        self.convert_embedding_to_bytes = convert_embedding_to_bytes
        self.convert_bytes_to_embedding = convert_bytes_to_embedding


semantic_service = SemanticService()
