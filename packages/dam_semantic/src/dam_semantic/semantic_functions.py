import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from dam.core.transaction import EcsTransaction
from dam.models.core.entity import Entity

# Import necessary tag components and functions to fetch tags
from dam_sire.resource import SireResource
from sentence_transformers import SentenceTransformer

from .models import (
    BaseSpecificEmbeddingComponent,
    ModelHyperparameters,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_PARAMS: ModelHyperparameters = {}

SENTENCE_TRANSFORMER_IDENTIFIER = "sentence_transformer"


def _load_sentence_transformer_model_sync(
    model_name_or_path: str, params: Optional[Dict[str, Any]] = None
) -> SentenceTransformer:
    logger.info(f"Attempting to load SentenceTransformer model: {model_name_or_path} with params {params}")
    if params:
        device = params.pop("device", None)
    else:
        device = None
    model = SentenceTransformer(model_name_or_path, device=device, **(params or {}))
    logger.info(f"SentenceTransformer model {model_name_or_path} loaded successfully.")
    return model


async def generate_embedding(
    sire_resource: "SireResource",
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    params: Optional[ModelHyperparameters] = None,
) -> Optional[np.ndarray]:
    if not text or not text.strip():
        return None
    try:
        if params:
            params.pop("dimensions", None)
        model = SentenceTransformer(model_name, **(params or {}))
        with sire_resource.auto_manage(model) as managed_model_wrapper:
            managed_model = managed_model_wrapper.user.manage_object
            embedding = managed_model.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}", exc_info=True)
        return None


def convert_embedding_to_bytes(embedding: np.ndarray) -> bytes:
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return embedding.tobytes()


def convert_bytes_to_embedding(embedding_bytes: bytes, dtype: Any = np.float32) -> np.ndarray:
    return np.frombuffer(embedding_bytes, dtype=dtype)


class BatchTextItem(TypedDict):
    component_name: str
    field_name: str
    text_content: str


async def update_text_embeddings_for_entity(
    transaction: EcsTransaction,
    entity_id: int,
    text_fields_map: Dict[str, Any],
    sire_resource: "SireResource",
    model_name: str = DEFAULT_MODEL_NAME,
    model_params: Optional[ModelHyperparameters] = None,
    batch_texts: Optional[List[BatchTextItem]] = None,
    include_manual_tags: bool = True,
    include_model_tags_config: Optional[List[Dict[str, Any]]] = None,
    tag_concatenation_strategy: str = " [TAGS] {tags_string}",
) -> List[BaseSpecificEmbeddingComponent]:
    logger.warning("update_text_embeddings_for_entity is not fully implemented with sire yet.")
    return []


async def find_similar_entities_by_text_embedding(
    transaction: EcsTransaction,
    query_text: str,
    sire_resource: "SireResource",
    model_name: str,
    model_params: Optional[ModelHyperparameters] = None,
    top_n: int = 10,
) -> List[Tuple[Entity, float, BaseSpecificEmbeddingComponent]]:
    logger.warning("find_similar_entities_by_text_embedding is not fully implemented with sire yet.")
    return []
