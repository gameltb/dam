import numpy as np
import pytest

# ModelExecutionManager fixture from conftest.py
from dam_semantic import semantic_functions

# Models to be tested (actual specific component classes)
from dam_semantic.models import (
    ModelHyperparameters,
    get_embedding_component_class,
)

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
async def test_generate_embedding_and_conversion(sire_resource):
    text = "Hello world"
    # Test with MiniLM
    embedding_minilm_np = await semantic_functions.generate_embedding(
        sire_resource,
        text,
        model_name=TEST_MODEL_MINILM,
        params=TEST_PARAMS_MINILM,
    )
    assert embedding_minilm_np is not None
    assert isinstance(embedding_minilm_np, np.ndarray)
    assert embedding_minilm_np.shape[0] == 384

    # Test with CLIP
    embedding_clip_np = await semantic_functions.generate_embedding(
        sire_resource,
        text,
        model_name=TEST_MODEL_CLIP,
        params=TEST_PARAMS_CLIP,
    )
    assert embedding_clip_np is not None
    assert isinstance(embedding_clip_np, np.ndarray)
    assert embedding_clip_np.shape[0] == 512

    assert not np.array_equal(embedding_minilm_np, embedding_clip_np)

    assert (
        await semantic_functions.generate_embedding(
            sire_resource, "", model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM
        )
        is None
    )
    assert (
        await semantic_functions.generate_embedding(
            sire_resource, "   ", model_name=TEST_MODEL_MINILM, params=TEST_PARAMS_MINILM
        )
        is None
    )

    embedding_bytes = semantic_functions.convert_embedding_to_bytes(embedding_minilm_np)
    assert isinstance(embedding_bytes, bytes)
    assert len(embedding_bytes) == 384 * 4

    embedding_np_restored = semantic_functions.convert_bytes_to_embedding(embedding_bytes)
    assert np.array_equal(embedding_minilm_np, embedding_np_restored)
