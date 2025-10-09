"""Defines audio-related functions, including mock model behavior for testing."""

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
    """A mock audio model for testing purposes."""

    def __init__(self, model_name: str, params: AudioModelHyperparameters | None) -> None:
        """Initialize the mock audio model."""
        self.model_name = model_name
        self.params = params
        self.output_dim = (
            AUDIO_EMBEDDING_MODEL_REGISTRY.get(model_name, {}).get("default_params", {}).get("dimensions", 128)
        )
        logger.info("MockAudioModel '%s' initialized with output_dim: %s", model_name, self.output_dim)

    def encode(self, audio_path: str, **_kwargs: Any) -> np.ndarray:
        """Simulate encoding an audio file."""
        logger.info("MockAudioModel '%s': Simulating encoding for '%s'", self.model_name, audio_path)
        return np.random.rand(self.output_dim).astype(np.float32)

    async def encode_async(self, audio_path: str, **kwargs: Any) -> np.ndarray:
        """Asynchronously encode an audio file."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.encode, audio_path, **kwargs)


async def get_mock_audio_model(  # type: ignore[no-any-unimported]
    sire_resource: "SireResource",  # noqa: ARG001
    model_name: str = DEFAULT_AUDIO_MODEL_NAME,  # noqa: ARG001
    params: AudioModelHyperparameters | None = None,  # noqa: ARG001
) -> Optional["AutoManageWrapper[MockAudioModel]"]:
    """Get a mock audio model."""
    AutoManageWrapper.register_type_wrapper(MockAudioModel, TorchModuleWrapper)
    # return sire_resource.get_model(MockAudioModel, model_name, params=params)
    return None


def convert_embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Convert a numpy float32 embedding to bytes."""
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    return embedding.tobytes()


def convert_bytes_to_embedding(embedding_bytes: bytes, dtype: Any = np.float32) -> np.ndarray:
    """Convert bytes back to a numpy embedding."""
    return np.frombuffer(embedding_bytes, dtype=dtype)


async def generate_audio_embedding_for_entity(  # type: ignore[no-any-unimported]
    transaction: WorldTransaction,  # noqa: ARG001
    sire_resource: "SireResource",  # noqa: ARG001
    entity_id: int,  # noqa: ARG001
    model_name: str = DEFAULT_AUDIO_MODEL_NAME,  # noqa: ARG001
    model_params: AudioModelHyperparameters | None = None,  # noqa: ARG001
    audio_file_path: str | None = None,  # noqa: ARG001
) -> BaseSpecificAudioEmbeddingComponent | None:
    """
    Generate an audio embedding for a given entity.

    Note: This function is not fully implemented yet.
    """
    logger.warning("generate_audio_embedding_for_entity is not fully implemented with sire yet.")
    return None


async def find_similar_entities_by_audio_embedding(  # type: ignore[no-any-unimported]
    transaction: WorldTransaction,  # noqa: ARG001
    sire_resource: "SireResource",  # noqa: ARG001
    query_audio_path: str,  # noqa: ARG001
    model_name: str,  # noqa: ARG001
    model_params: AudioModelHyperparameters | None = None,  # noqa: ARG001
    top_n: int = 10,  # noqa: ARG001
) -> list[tuple[Entity, float, BaseSpecificAudioEmbeddingComponent]]:
    """
    Find similar entities by audio embedding.

    Note: This function is not fully implemented yet.
    """
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
