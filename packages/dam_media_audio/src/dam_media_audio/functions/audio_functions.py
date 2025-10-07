import asyncio
import logging
from typing import Any, Optional

import numpy as np
from dam.core.transaction import WorldTransaction
from dam.models.core.entity import Entity
from dam_semantic.models.audio_embedding_component import (
    AUDIO_EMBEDDING_MODEL_REGISTRY,
    AudioModelHyperparameters,
    BaseSpecificAudioEmbeddingComponent,
)
from dam_sire.resource import SireResource
from sire.core.runtime_resource_management import AutoManageWrapper
from sire.core.runtime_resource_user.pytorch_module import TorchModuleWrapper

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_MODEL_NAME = "vggish"
MOCK_AUDIO_MODEL_IDENTIFIER = "mock_audio_model"


class MockAudioModel:
    def __init__(self, model_name: str, params: AudioModelHyperparameters | None) -> None:
        self.model_name = model_name
        self.params = params
        self.output_dim = (
            AUDIO_EMBEDDING_MODEL_REGISTRY.get(model_name, {}).get("default_params", {}).get("dimensions", 128)
        )
        logger.info(f"MockAudioModel '{model_name}' initialized with output_dim: {self.output_dim}")

    def encode(self, audio_path: str, **kwargs: Any) -> np.ndarray:
        logger.info(f"MockAudioModel '{self.model_name}': Simulating encoding for '{audio_path}'")
        return np.random.rand(self.output_dim).astype(np.float32)

    async def encode_async(self, audio_path: str, **kwargs: Any) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.encode, audio_path, **kwargs)


async def get_mock_audio_model(  # type: ignore[no-any-unimported]
    sire_resource: "SireResource",
    model_name: str = DEFAULT_AUDIO_MODEL_NAME,
    params: AudioModelHyperparameters | None = None,
) -> Optional["AutoManageWrapper[MockAudioModel]"]:
    AutoManageWrapper.register_type_wrapper(MockAudioModel, TorchModuleWrapper)
    # return sire_resource.get_model(MockAudioModel, model_name, params=params)
    return None


def convert_embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Converts a numpy float32 embedding to bytes."""
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return embedding.tobytes()


def convert_bytes_to_embedding(embedding_bytes: bytes, dtype: Any = np.float32) -> np.ndarray:
    """Converts bytes back to a numpy embedding."""
    return np.frombuffer(embedding_bytes, dtype=dtype)


async def generate_audio_embedding_for_entity(  # type: ignore[no-any-unimported]
    transaction: WorldTransaction,
    sire_resource: "SireResource",
    entity_id: int,
    model_name: str = DEFAULT_AUDIO_MODEL_NAME,
    model_params: AudioModelHyperparameters | None = None,
    audio_file_path: str | None = None,
) -> BaseSpecificAudioEmbeddingComponent | None:
    logger.warning("generate_audio_embedding_for_entity is not fully implemented with sire yet.")
    return None


async def find_similar_entities_by_audio_embedding(  # type: ignore[no-any-unimported]
    transaction: WorldTransaction,
    sire_resource: "SireResource",
    query_audio_path: str,
    model_name: str,
    model_params: AudioModelHyperparameters | None = None,
    top_n: int = 10,
) -> list[tuple[Entity, float, BaseSpecificAudioEmbeddingComponent]]:
    logger.warning("find_similar_entities_by_audio_embedding is not fully implemented with sire yet.")
    return []


__all__ = [
    "DEFAULT_AUDIO_MODEL_NAME",
    "convert_bytes_to_embedding",
    "convert_embedding_to_bytes",
    "find_similar_entities_by_audio_embedding",
    "generate_audio_embedding_for_entity",
    "get_mock_audio_model",
]
