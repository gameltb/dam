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


@pytest.fixture
def model_manager():
    # To ensure tests are isolated, we can mock the device manager
    # to avoid actual hardware detection during tests.
    from unittest.mock import patch

    from sire import ModelManager

    with patch("sire.manager.DeviceManager") as MockDeviceManager:
        mock_device_manager = MockDeviceManager.return_value
        mock_device_manager.select_device.return_value = "cpu"  # Default mock behavior

        manager = ModelManager()
        manager._device_manager = mock_device_manager
        yield manager
