from abc import ABC, abstractmethod
from typing import Any

import torch


class InferenceRuntime(ABC):
    @abstractmethod
    def load(self, model_path: str, device: str, **kwargs) -> Any: ...

    @abstractmethod
    def predict(self, model: Any, data: Any) -> Any: ...

    @abstractmethod
    def get_memory_footprint(self, model: Any) -> int: ...


class PyTorchRuntime(InferenceRuntime):
    def load(self, model_path: str, device: str, **kwargs) -> Any:
        model_class = kwargs.get("model_class")
        if not model_class:
            raise ValueError("`model_class` must be provided for PyTorch runtime.")

        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    def predict(self, model: Any, data: Any) -> Any:
        with torch.no_grad():
            if isinstance(data, (list, tuple)):
                return model(*data)
            elif isinstance(data, dict):
                return model(**data)
            else:
                return model(data)

    def get_memory_footprint(self, model: Any) -> int:
        mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
        mem = mem_params + mem_bufs
        return mem
