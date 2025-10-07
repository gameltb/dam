"""Fixtures for the sire tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn


@pytest.fixture(scope="session")
def temp_model_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class SimpleTestModel(nn.Module):
    """A simple model for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model."""
        super().__init__(*args, **kwargs)  # type: ignore
        self.linear = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass."""
        return self.linear(x)


@pytest.fixture(scope="module")
def simple_model_file(temp_model_dir: str) -> str:
    """
    Create a simple model and save it to a file.

    Args:
        temp_model_dir: The temporary directory to save the model in.

    Returns:
        The path to the saved model file.

    """
    model = SimpleTestModel()
    model_path = Path(temp_model_dir) / "simple_test_model.pth"
    torch.save(model.state_dict(), model_path)
    return str(model_path)
