import torch

from sire.runtimes import PyTorchRuntime

from .conftest import SimpleTestModel


def test_pytorch_runtime_load(simple_model_file: str):
    runtime = PyTorchRuntime()
    model = runtime.load(model_path=simple_model_file, device="cpu", model_class=SimpleTestModel)
    assert isinstance(model, SimpleTestModel)


def test_pytorch_runtime_predict(simple_model_file: str):
    runtime = PyTorchRuntime()
    model = runtime.load(model_path=simple_model_file, device="cpu", model_class=SimpleTestModel)
    dummy_data = torch.randn(1, 5)
    result = runtime.predict(model, dummy_data)
    assert result.shape == (1, 2)


def test_pytorch_runtime_get_memory_footprint(simple_model_file: str):
    runtime = PyTorchRuntime()
    model = runtime.load(model_path=simple_model_file, device="cpu", model_class=SimpleTestModel)
    # A simple check to ensure it returns a non-zero integer
    assert isinstance(runtime.get_memory_footprint(model), int)
    assert runtime.get_memory_footprint(model) > 0
