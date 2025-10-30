"""Fixtures for the `dam_semantic` package tests."""

from typing import Any

import pytest
import torch
from dam_sire.resource import SireResource
from pytest_mock import MockerFixture

pytest_plugins = ["dam_test_utils.fixtures"]


@pytest.fixture
def sire_resource() -> SireResource:
    """Return a SireResource."""
    return SireResource()


@pytest.fixture
def mock_transformer_model(mocker: MockerFixture) -> tuple[Any, Any]:
    """Mock the AutoModel and AutoTokenizer."""
    mock_tokenizer = mocker.patch("transformers.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("transformers.AutoModel.from_pretrained")

    mock_tokenizer.return_value.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    return mock_model, mock_tokenizer
