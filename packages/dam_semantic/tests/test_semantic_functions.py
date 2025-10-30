"""Tests for the semantic functions."""

from typing import Any

import numpy as np
import pytest
import torch
from dam_sire.resource import SireResource
from pytest_mock import MockerFixture

from dam_semantic import semantic_functions
from dam_semantic.models.text_embedding_component import (
    ModelHyperparameters,
)

TEST_MODEL_MINILM = "all-MiniLM-L6-v2"
TEST_PARAMS_MINILM: ModelHyperparameters = {"trust_remote_code": True}


def test_mean_pooling() -> None:
    """Test the mean pooling function."""
    model_output = (torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),)
    attention_mask = torch.tensor([[1, 1]])
    result = semantic_functions.mean_pooling(model_output, attention_mask)
    expected = torch.tensor([[2.0, 3.0]])
    assert torch.allclose(result, expected)


@pytest.mark.asyncio
async def test_generate_embedding_and_conversion(
    mock_transformer_model: Any,
    mocker: MockerFixture,
) -> None:
    """Test the generation and conversion of embeddings."""
    mock_model, mock_tokenizer = mock_transformer_model
    mock_model.return_value.parameters.return_value = [torch.nn.Parameter(torch.randn(1, 1))]
    mock_mean_pooling = mocker.patch("dam_semantic.semantic_functions.mean_pooling")
    mock_mean_pooling.return_value = torch.randn(1, 384)
    mocker.patch("dam_semantic.semantic_functions.torch.no_grad")

    # Mock the auto_manage context manager
    mock_managed_model_wrapper = mocker.MagicMock()
    mock_managed_model_wrapper.get_manage_object.return_value = mock_model.return_value

    async_magic_mock = mocker.AsyncMock()
    async_magic_mock.__aenter__.return_value = mock_managed_model_wrapper
    mocker.patch("dam_sire.resource.SireResource.auto_manage", return_value=async_magic_mock)

    sire_resource = SireResource()
    text = "Hello world"
    embedding_minilm_np = await semantic_functions.generate_embedding(
        sire_resource,
        text,
        model_name=TEST_MODEL_MINILM,
        params=TEST_PARAMS_MINILM,
    )

    mock_tokenizer.assert_called_with(TEST_MODEL_MINILM)
    mock_model.assert_called_with(TEST_MODEL_MINILM, **TEST_PARAMS_MINILM)
    mock_mean_pooling.assert_called_once()

    assert embedding_minilm_np is not None
    assert isinstance(embedding_minilm_np, np.ndarray)
    assert embedding_minilm_np.shape[0] == 384

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
    np.testing.assert_allclose(embedding_minilm_np, embedding_np_restored, atol=1e-6)
