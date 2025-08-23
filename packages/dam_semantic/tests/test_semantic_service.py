from typing import Any, List, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

# ModelExecutionManager fixture from conftest.py
from dam.core.model_manager import ModelExecutionManager

# Models to be tested (actual specific component classes)
from dam_semantic.models import (
    ModelHyperparameters,
    get_embedding_component_class,
)
from dam_semantic import service as semantic_service
from dam.services import ecs_service
from dam_semantic.service import BatchTextItem  # Keep this for type hints

class MockSentenceTransformer:
    def __init__(self, model_name_or_path=None, **kwargs):
        self.model_name = model_name_or_path
        if model_name_or_path and "clip" in model_name_or_path.lower():
            self.dim = 512
        elif model_name_or_path and "MiniLM-L6-v2" in model_name_or_path:
            self.dim = 384
        else:
            self.dim = 384

    def encode(self, sentences, convert_to_numpy=True, **kwargs):
        original_sentences_type = type(sentences)
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = []
        for s in sentences:
            if not s or not s.strip():
                embeddings.append(np.zeros(self.dim, dtype=np.float32))
                continue
            sum_ords = sum(ord(c) for c in s)
            model_ord_sum = sum(ord(c) for c in (self.model_name or "default"))
            vec_elements = [sum_ords % 100, len(s) % 100, model_ord_sum % 100]
            if self.dim >= 3:
                vec = np.array(vec_elements[: self.dim] + [0.0] * (self.dim - min(3, self.dim)), dtype=np.float32)
            elif self.dim > 0:
                vec = np.array(vec_elements[: self.dim], dtype=np.float32)
            else:
                vec = np.array([], dtype=np.float32)
            if vec.shape[0] != self.dim and self.dim > 0:
                padding = np.zeros(self.dim - vec.shape[0], dtype=np.float32)
                vec = np.concatenate((vec, padding))
            elif vec.shape[0] != self.dim and self.dim == 0:
                vec = np.array([], dtype=np.float32)
            embeddings.append(vec)
        if not convert_to_numpy:
            embeddings = [e.tolist() for e in embeddings]
        if original_sentences_type is str:
            return embeddings[0] if embeddings else np.array([])
        else:
            return np.array(embeddings) if convert_to_numpy else embeddings

# Define model names and params to be used in tests, corresponding to registered models
TEST_MODEL_MINILM = "all-MiniLM-L6-v2"
TEST_PARAMS_MINILM: ModelHyperparameters = {"dimensions": 384}  # Must match registry

TEST_MODEL_CLIP = "clip-ViT-B-32"
TEST_PARAMS_CLIP: ModelHyperparameters = {"dimensions": 512}  # Must match registry

# Get the component classes for easier use in tests
MiniLMEmbeddingComponent = get_embedding_component_class(TEST_MODEL_MINILM, TEST_PARAMS_MINILM)
ClipEmbeddingComponent = get_embedding_component_class(TEST_MODEL_CLIP, TEST_PARAMS_CLIP)

# Ensure the test models are actually registered and component classes are found
assert MiniLMEmbeddingComponent is not None, "Test MiniLM model not registered or class not found"
assert ClipEmbeddingComponent is not None, "Test CLIP model not registered or class not found"

@pytest.mark.asyncio
async def test_generate_embedding_and_conversion(
    global_model_execution_manager: ModelExecutionManager,
):
    text = "Hello world"
    # Test with MiniLM
    embedding_minilm_np = await semantic_service.generate_embedding(
        global_model_execution_manager,
        text,
        model_name=TEST_MODEL_MINILM,
        params=TEST_PARAMS_MINILM,
    )
    assert embedding_minilm_np is not None
    assert isinstance(embedding_minilm_np, np.ndarray)
    assert embedding_minilm_np.shape == (TEST_PARAMS_MINILM["dimensions"],)

    # Test with CLIP
    embedding_clip_np = await semantic_service.generate_embedding(
        global_model_execution_manager,
        text,
        model_name=TEST_MODEL_CLIP,
        params=TEST_PARAMS_CLIP,
    )
    assert embedding_clip_np is not None
    assert isinstance(embedding_clip_np, np.ndarray)
    assert embedding_clip_np.shape == (TEST_PARAMS_CLIP["dimensions"],)

    assert not np.array_equal(embedding_minilm_np, embedding_clip_np)

    assert (
        await semantic_service.generate_embedding(
            global_model_execution_manager, "", model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM
        )
        is None
    )
    assert (
        await semantic_service.generate_embedding(
            global_model_execution_manager, "   ", model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM
        )
        is None
    )

    embedding_bytes = semantic_service.convert_embedding_to_bytes(embedding_minilm_np)
    assert isinstance(embedding_bytes, bytes)
    assert len(embedding_bytes) == TEST_PARAMS_MINILM["dimensions"] * 4

    embedding_np_restored = semantic_service.convert_bytes_to_embedding(embedding_bytes)
    assert np.array_equal(embedding_minilm_np, embedding_np_restored)

    unregistered_embedding = await semantic_service.generate_embedding(
        global_model_execution_manager, "test", model_name="unregistered-model", params={}
    )
    assert unregistered_embedding is not None
