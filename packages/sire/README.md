# Sire

A dynamic, just-in-time resource manager for PyTorch models, enabling efficient execution on constrained hardware.

## Overview

Sire is a Python library designed to manage PyTorch models in memory-constrained environments, such as a local machine with a single GPU. It allows you to work with more models than can fit into your GPU's VRAM at once by dynamically loading and offloading them between GPU and CPU memory on demand.

The core principle is **just-in-time resource allocation**. A model is only moved to the GPU right before it's used for inference and is moved back to the CPU (or disk) immediately after, freeing up VRAM for other models.

## Key Features

- **Automatic GPU/CPU Swapping**: Models are automatically moved to the runtime device (e.g., GPU) when needed and offloaded afterward.
- **`auto_manage` Context Manager**: A simple `with` statement to handle resource management for a block of code.
- **`AutoManageHook`**: Attach to a `torch.nn.Module` to make it self-managing during its `forward` pass.
- **Extensible**: Built on a flexible system of Resource Pools and Users that can be extended.
- **Commit-based State Management**: An advanced feature allowing you to apply and revert patches (like LoRAs or function hooks) to models as "commits".

## Quick Start

Here is a simple example of how to use Sire to manage a PyTorch model.

```python
import torch
import sire

# 1. Initialize Sire's resource pools
sire.setup_default_pools()

# 2. Create your PyTorch model
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        print(f"Executing forward on device: {x.device}")
        return self.linear(x)

model = MyModel()
print(f"Model initial device: {next(model.parameters()).device}")

# 3. Wrap the model to make it Sire-aware
managed_model = sire.manage(model)

# This is only needed for demonstration if you don't have a CUDA device.
# In a real GPU scenario, this would be detected automatically.
if not torch.cuda.is_available():
    print("No CUDA device found, model will stay on CPU.")
else:
    # 4. Use the auto_manage context manager for inference
    print("\nEntering auto_manage context...")
    with sire.auto_manage(managed_model) as am_wrapper:
        # Inside this block, the model is moved to the GPU
        execution_device = am_wrapper.get_execution_device()
        print(f"Model is now on device: {next(model.parameters()).device}")

        # Input tensor must be on the same device
        dummy_input = torch.randn(1, 10).to(execution_device)

        # Run inference
        output = model(dummy_input)

    print("Exited auto_manage context.")
    print(f"Model is now back on device: {next(model.parameters()).device}")
```

## How It Works

1.  **`sire.setup_default_pools()`**: This function scans for available devices (CPU and CUDA) and creates a `ResourcePool` for each one.
2.  **`sire.manage(model)`**: This wraps your model in an `AutoManageWrapper`, which makes it a "resource user". The wrapper automatically selects the correct backend (e.g., `TorchModuleWrapper` for `nn.Module`) and registers it with Sire's central `ResourcePoolManagement` singleton.
3.  **`with sire.auto_manage(...)`**: When you enter this context, the manager ensures the model's required memory is available in the designated "runtime" pool (e.g., the GPU pool). It moves the model to the GPU. When you exit the context, the model is offloaded back to its "offload" pool (the CPU pool).
