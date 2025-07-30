# Sire - Design Specification

## 1. Overview

Sire is a lightweight, single-machine PyTorch model scheduling library designed for simplicity and ease of use. It provides a high-level API to manage the lifecycle of models, automate resource allocation (especially GPU memory), and run inference, while abstracting away the complexities of the underlying hardware and runtimes.

## 2. Core Components

The library will be built around a few key components:

- **`ModelManager`**: The main user-facing API for all operations.
- **`DeviceManager`**: A component for managing and monitoring hardware devices (CPUs and GPUs).
- **`InferenceRuntime`**: An abstraction for different model execution backends (e.g., PyTorch, ONNX).

## 3. API Design

### 3.1. `ModelManager`

The `ModelManager` is the central entry point for users.

**Example Usage:**

```python
from sire import ModelManager

# Initialize the manager
manager = ModelManager()

# Register a model
manager.register_model(
    name="my_classifier",
    model_path="path/to/model.pth",
    runtime="pytorch",
    model_class=MyModelClass, # For PyTorch models
)

# Load the model into memory (auto-selects device)
manager.load_model("my_classifier")

# Run inference
input_data = ...
result = manager.predict("my_classifier", input_data)

# Unload the model to free up memory
manager.unload_model("my_classifier")
```

**Methods:**

- `register_model(name: str, model_path: str, runtime: str, **kwargs)`: Registers a model with the manager. The `runtime` parameter specifies the backend to use (e.g., "pytorch", "onnx"). `kwargs` can contain runtime-specific information, like a `model_class` for PyTorch models.
- `load_model(name: str, device: str = "auto")`: Loads a registered model onto a device. If `device` is "auto", the `DeviceManager` will select the best available device.
- `unload_model(name: str)`: Unloads a model from memory.
- `predict(model_name: str, data: Any) -> Any`: Runs inference on a loaded model.
- `get_model_stats(name: str) -> dict`: Returns performance metrics for a model (e.g., inference count, average latency, memory usage).

### 3.2. `DeviceManager`

The `DeviceManager` is responsible for abstracting hardware resources.

**Responsibilities:**

- Detect available devices (CPUs and GPUs).
- Monitor memory usage on each device.
- Select the optimal device for model loading when requested by the `ModelManager`.

### 3.3. `InferenceRuntime` (Pluggable Runtimes)

To support different backends, we will use a strategy pattern. An abstract base class `InferenceRuntime` will define the interface for runtimes.

**Interface (`InferenceRuntime`):**

```python
from abc import ABC, abstractmethod

class InferenceRuntime(ABC):
    @abstractmethod
    def load(self, model_path: str, **kwargs) -> Any:
        ...

    @abstractmethod
    def predict(self, model: Any, data: Any) -> Any:
        ...

    @abstractmethod
    def get_memory_footprint(self, model: Any) -> int:
        ...
```

Concrete implementations like `PyTorchRuntime` and `ONNXRuntime` will implement this interface. The `ModelManager` will instantiate the appropriate runtime when a model is registered.

## 4. GPU Memory Management

The `ModelManager`, in conjunction with the `DeviceManager`, will manage GPU memory automatically.

- **On-demand Loading:** Models are only loaded into memory when `load_model` is called.
- **Automatic Placement:** When `load_model` is called with `device="auto"`, the `DeviceManager` will find a GPU with sufficient memory.
- **LRU Cache (Future):** A potential future enhancement is an LRU (Least Recently Used) cache policy to automatically unload models when GPU memory is scarce.

## 5. Multi-GPU Scheduling

For a single-machine, multi-GPU setup, the `DeviceManager` will provide abstractions for simple load balancing.

- **Device Selection:** The `DeviceManager` can be configured with a strategy for device selection (e.g., "round-robin", "least-memory-used").
- **Simplified API:** The user will not need to specify device IDs unless they want to override the automatic behavior.

## 6. Monitoring and Error Handling

- **Monitoring:** The `ModelManager` will collect basic metrics. These can be accessed via `get_model_stats`.
- **Error Handling:** The library will use custom exceptions to provide clear error messages for common issues (e.g., `ModelNotFound`, `InsufficientMemory`, `InferenceError`).
