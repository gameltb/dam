import os
import tempfile

import pytest
import torch
import torch.nn as nn


# Using a temporary directory for test artifacts
@pytest.fixture(scope="session")
def temp_model_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture(scope="module")
def simple_model_file(temp_model_dir):
    model = SimpleTestModel()
    model_path = os.path.join(temp_model_dir, "simple_test_model.pth")
    torch.save(model.state_dict(), model_path)
    return model_path


