import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dam.models.core.entity import Entity
from dam.models.semantic.audio_embedding_component import (
    AUDIO_EMBEDDING_MODEL_REGISTRY,
    AudioModelHyperparameters,
    BaseSpecificAudioEmbeddingComponent,
    get_audio_embedding_component_class,
)
from dam.services import ecs_service

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_MODEL_NAME = "vggish"
MOCK_AUDIO_MODEL_IDENTIFIER = "mock_audio_model"


def convert_embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Converts a numpy float32 embedding to bytes."""
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return embedding.tobytes()


def convert_bytes_to_embedding(embedding_bytes: bytes, dtype=np.float32) -> np.ndarray:
    """Converts bytes back to a numpy embedding."""
    return np.frombuffer(embedding_bytes, dtype=dtype)


__all__ = [
    "convert_embedding_to_bytes",
    "convert_bytes_to_embedding",
    "DEFAULT_AUDIO_MODEL_NAME",
]
