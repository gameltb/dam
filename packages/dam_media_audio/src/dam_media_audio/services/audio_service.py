import logging
from typing import Any, Dict, List, Optional, Tuple
import asyncio

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
from dam_sire.resource import SireResource
from sire.core.runtime_resource_management import AutoManageWrapper
from sire.core.runtime_resource_user.pytorch_module import TorchModuleWrapper

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_MODEL_NAME = "vggish"
MOCK_AUDIO_MODEL_IDENTIFIER = "mock_audio_model"


class MockAudioModel:
    def __init__(self, model_name: str, params: Optional[AudioModelHyperparameters]):
        self.model_name = model_name
        self.params = params
        self.output_dim = (
            AUDIO_EMBEDDING_MODEL_REGISTRY.get(model_name, {}).get("default_params", {}).get("dimensions", 128)
        )
        logger.info(f"MockAudioModel '{model_name}' initialized with output_dim: {self.output_dim}")

    def encode(self, audio_path: str, **kwargs) -> np.ndarray:
        logger.info(f"MockAudioModel '{self.model_name}': Simulating encoding for '{audio_path}'")
        return np.random.rand(self.output_dim).astype(np.float32)

    async def encode_async(self, audio_path: str, **kwargs) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.encode, audio_path, **kwargs)


def _load_mock_audio_model_sync(model_name_or_path: str, params: Optional[Dict[str, Any]] = None) -> MockAudioModel:
    return MockAudioModel(model_name_or_path, params)


async def get_mock_audio_model(
    sire_resource: "SireResource",
    model_name: str = DEFAULT_AUDIO_MODEL_NAME,
    params: Optional[AudioModelHyperparameters] = None,
) -> "AutoManageWrapper":
    AutoManageWrapper.registe_type_wrapper(MockAudioModel, TorchModuleWrapper)
    return sire_resource.get_model(MockAudioModel, model_name, params=params)


def convert_embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Converts a numpy float32 embedding to bytes."""
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return embedding.tobytes()


def convert_bytes_to_embedding(embedding_bytes: bytes, dtype=np.float32) -> np.ndarray:
    """Converts bytes back to a numpy embedding."""
    return np.frombuffer(embedding_bytes, dtype=dtype)


async def generate_audio_embedding_for_entity(
    session: AsyncSession,
    sire_resource: "SireResource",
    entity_id: int,
    model_name: str = DEFAULT_AUDIO_MODEL_NAME,
    model_params: Optional[AudioModelHyperparameters] = None,
    audio_file_path: Optional[str] = None,
) -> Optional[BaseSpecificAudioEmbeddingComponent]:
    logger.warning("generate_audio_embedding_for_entity is not fully implemented with sire yet.")
    return None


async def find_similar_entities_by_audio_embedding(
    session: AsyncSession,
    sire_resource: "SireResource",
    query_audio_path: str,
    model_name: str,
    model_params: Optional[AudioModelHyperparameters] = None,
    top_n: int = 10,
) -> List[Tuple[Entity, float, BaseSpecificAudioEmbeddingComponent]]:
    logger.warning("find_similar_entities_by_audio_embedding is not fully implemented with sire yet.")
    return []


__all__ = [
    "convert_embedding_to_bytes",
    "convert_bytes_to_embedding",
    "DEFAULT_AUDIO_MODEL_NAME",
    "get_mock_audio_model",
    "generate_audio_embedding_for_entity",
    "find_similar_entities_by_audio_embedding",
]
