import pytest
import torch

from sire import ModelManager, ModelNotFoundError, ModelNotLoadedError

from .conftest import SimpleTestModel


def test_register_model(model_manager: ModelManager, simple_model_file: str):
    model_manager.register_model(
        name="test_model", model_path=simple_model_file, runtime="pytorch", model_class=SimpleTestModel
    )
    assert "test_model" in model_manager._models


def test_register_duplicate_model_raises_error(model_manager: ModelManager, simple_model_file: str):
    model_manager.register_model(
        name="test_model", model_path=simple_model_file, runtime="pytorch", model_class=SimpleTestModel
    )
    with pytest.raises(ValueError):
        model_manager.register_model(
            name="test_model", model_path=simple_model_file, runtime="pytorch", model_class=SimpleTestModel
        )


def test_load_and_unload_model(model_manager: ModelManager, simple_model_file: str):
    model_manager.register_model(
        name="test_model", model_path=simple_model_file, runtime="pytorch", model_class=SimpleTestModel
    )

    # Load
    model_manager.load_model("test_model", device="cpu")
    stats = model_manager.get_model_stats("test_model")
    assert stats["loaded"] is True
    assert stats["device"] == "cpu"

    # Unload
    model_manager.unload_model("test_model")
    stats = model_manager.get_model_stats("test_model")
    assert stats["loaded"] is False
    assert stats["device"] is None


def test_predict(model_manager: ModelManager, simple_model_file: str):
    model_manager.register_model(
        name="test_model", model_path=simple_model_file, runtime="pytorch", model_class=SimpleTestModel
    )
    model_manager.load_model("test_model", device="cpu")

    dummy_data = torch.randn(1, 5)
    result = model_manager.predict("test_model", dummy_data)
    assert result.shape == (1, 2)


def test_predict_on_unloaded_model_raises_error(model_manager: ModelManager, simple_model_file: str):
    model_manager.register_model(
        name="test_model", model_path=simple_model_file, runtime="pytorch", model_class=SimpleTestModel
    )
    with pytest.raises(ModelNotLoadedError):
        model_manager.predict("test_model", torch.randn(1, 5))


def test_load_unregistered_model_raises_error(model_manager: ModelManager):
    with pytest.raises(ModelNotFoundError):
        model_manager.load_model("non_existent_model")
