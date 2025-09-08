import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest
import torch
import torch.nn as nn


# Using a temporary directory for test artifacts
@pytest.fixture(scope="session")
def temp_model_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class SimpleTestModel(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.linear = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture(scope="module")
def simple_model_file(temp_model_dir: Path) -> str:
    model = SimpleTestModel()
    model_path = os.path.join(str(temp_model_dir), "simple_test_model.pth")
    torch.save(model.state_dict(), model_path)
    return model_path
