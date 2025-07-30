from typing import Any, Dict

from .device import DeviceManager
from .exceptions import InferenceError, InsufficientMemoryError, ModelNotFoundError, ModelNotLoadedError
from .models import ModelConfig
from .runtimes import InferenceRuntime, PyTorchRuntime


class ModelManager:
    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._runtimes: Dict[str, InferenceRuntime] = {"pytorch": PyTorchRuntime()}
        self._device_manager = DeviceManager()

    def register_model(self, name: str, model_path: str, runtime: str, **kwargs):
        if name in self._models:
            raise ValueError(f"Model with name '{name}' is already registered.")

        config = ModelConfig(name=name, model_path=model_path, runtime=runtime, **kwargs)
        self._models[name] = config

    def load_model(self, name: str, device: str = "auto"):
        if name not in self._models:
            raise ModelNotFoundError(f"Model '{name}' not found.")

        config = self._models[name]
        runtime = self._runtimes.get(config.runtime)
        if not runtime:
            raise ValueError(f"Unsupported runtime: {config.runtime}")

        # This is a placeholder for memory estimation.
        # A more advanced implementation would be needed for accurate estimation.
        memory_required = 0

        if device == "auto":
            selected_device = self._device_manager.select_device(memory_required)
            if not selected_device:
                raise InsufficientMemoryError("No device with sufficient memory available.")
            config.device = selected_device
        else:
            config.device = device

        try:
            config.model = runtime.load(config.model_path, device=config.device, model_class=config.model_class)
        except Exception as e:
            config.device = None
            raise InferenceError(f"Failed to load model '{name}': {e}") from e

    def unload_model(self, name: str):
        if name not in self._models:
            raise ModelNotFoundError(f"Model '{name}' not found.")
        config = self._models[name]

        if config.device and config.device.startswith("cuda"):
            import torch

            # Clear model from GPU memory
            del config.model
            torch.cuda.empty_cache()

        config.model = None
        config.device = None

    def predict(self, model_name: str, data: Any) -> Any:
        if model_name not in self._models:
            raise ModelNotFoundError(f"Model '{model_name}' not found.")

        config = self._models[model_name]
        if not config.model or not config.device:
            raise ModelNotLoadedError(f"Model '{model_name}' is not loaded.")

        runtime = self._runtimes.get(config.runtime)
        if not runtime:
            raise ValueError(f"Unsupported runtime: {config.runtime}")

        try:
            return runtime.predict(config.model, data)
        except Exception as e:
            raise InferenceError(f"Inference failed for model '{model_name}': {e}") from e

    def get_model_stats(self, name: str) -> dict:
        if name not in self._models:
            raise ModelNotFoundError(f"Model '{name}' not found.")
        config = self._models[name]
        return {
            "name": config.name,
            "loaded": config.model is not None,
            "device": config.device,
            "runtime": config.runtime,
        }
