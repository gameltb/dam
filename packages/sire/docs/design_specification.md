# Sire - Design Specification

## 1. Overview

Sire is a dynamic, just-in-time resource manager for PyTorch models. Its primary goal is to enable the efficient execution of multiple models on a single machine with constrained hardware, particularly GPU memory. It achieves this by abstracting memory management and dynamically moving models between a high-speed "runtime" device (like a GPU) and a larger "offload" storage area (like CPU RAM).

The design is modular, decoupled, and built on a few core concepts that can be composed to handle simple and complex use cases.

## 2. Core Concepts

### 2.1. Resource Pools (`ResourcePool`)

A `ResourcePool` represents a source of memory on a specific device (e.g., a CUDA device or the CPU). It tracks its own available memory and manages a list of "Resource Users" that consume its memory.

### 2.2. Resource Users (`ResourcePoolUserABC`)

A "Resource User" is any object that consumes memory from a `ResourcePool`. The primary implementation is `TorchModuleWrapper`, designed to wrap a `torch.nn.Module`. A Resource User is responsible for:
- Reporting its current memory usage.
- Defining its "runtime" and "offload" memory pools.
- Handling being loaded to the runtime device (`on_load`).
- Handling being evicted to the offload device (`on_resource_request`).

### 2.3. Automatic Management (`auto_manage`)

This is the main user-facing tool, provided as a context manager.
- **`auto_manage(model, **kwargs)`**: This function wraps an object (like a `torch.nn.Module`) in an `AutoManageWrapper`, which selects the correct `ResourcePoolUser` implementation (e.g., `TorchModuleWrapper` for a module).
- **`__enter__`**: When the `with` block is entered, the wrapper requests the memory it needs. If the pool is full, it asks other, unlocked users to offload their resources. Once memory is secured, the target object is loaded.
- **`__exit__`**: When the block is exited, the wrapper is "unlocked", making it eligible for eviction.

### 2.4. Decoupled Components

The key to Sire's flexibility is its decoupled architecture. The core components are designed to be composed by the user.

#### `TorchModuleWrapper` - Flexible Memory Management
The `TorchModuleWrapper` is responsible *only* for memory management. It is completely agnostic to any optimization hooks that may be attached to the model it is managing. Its behavior can be configured via the `auto_manage` context manager:
- **`inference_memory_estimator`**: This argument to `auto_manage` controls how the wrapper estimates the additional VRAM needed for an inference pass. It can be:
    - An `int`: A fixed size in bytes.
    - A `callable`: A function with the signature `(model, user_context) -> int` that returns the required size.
    - A `Profiler` instance: The wrapper will invoke the profiler to automatically determine the required memory.
    - An `OptimizationPlan` instance: The wrapper will read the memory requirement from a pre-computed plan.

#### `InferenceOptimizerHook` - JIT Optimization
This is a powerful `ModelHook` that can be attached to a `torch.nn.Module` to dynamically optimize its execution. On the first forward pass, it runs a profiler, generates an `OptimizationPlan` to distribute the model across available devices, and re-hooks the model to execute this plan efficiently (e.g., by prefetching submodules from CPU to GPU just in time).

#### `CommitObjectProxy` - State Management
This is an advanced feature for managing different variations of a single base object (e.g., a model with different LoRAs applied) without deep copying, by applying and reverting "commits".

### 2.5. Hook Management (`HookManager`)

To manage the complexities of adding and removing hooks, especially for temporary operations like profiling, Sire provides a `HookManager`. Its primary feature is the `scope()` context manager:

```python
hook_manager = HookManager(model)
with hook_manager.scope():
    # The model has no hooks here.
    # Add temporary hooks for a specific task.
    add_hook_to_module(model, MyTemporaryHook())
    model(data)
# Outside the block, the original hooks are automatically restored.
```
This provides a standardized and safe paradigm for temporarily modifying a model's hook state, and is used internally by the `Profiler`.

## 3. Usage Patterns & Composition

The decoupled design allows for flexible composition to suit different needs.

### 3.1. Basic Management

Manage a simple model, letting Sire use a default memory estimation heuristic.

```python
import torch
from sire import auto_manage

model = torch.nn.Linear(100, 100)

with auto_manage(model):
    # Model is moved to the runtime device (e.g., GPU) here
    model(torch.randn(1, 100))
# Model is offloaded back to CPU here
```

### 3.2. Management with a Custom Memory Estimator

Provide a fixed integer value for the inference memory.

```python
# Reserve 1 GiB for inference
with auto_manage(model, inference_memory_estimator=1024**3):
    model(torch.randn(1, 100))
```

Or, provide a callable function to estimate memory based on input shape.

```python
def estimate_mem(model, context):
    input_tensor = context['args'][0]
    # Custom logic to calculate memory based on input shape
    return input_tensor.shape[0] * 1024 * 500

with auto_manage(model, inference_memory_estimator=estimate_mem):
    model(torch.randn(1, 100))
```

### 3.3. JIT Optimization with `InferenceOptimizerHook`

For large models that don't fit in VRAM, apply the `InferenceOptimizerHook` first, then let `auto_manage` handle the optimized model.

```python
from sire import InferenceOptimizerHook, add_hook_to_module

# 1. Create and configure the optimizer hook
optimizer_hook = InferenceOptimizerHook(cache_dir="my_opt_cache")

# 2. Apply the hook to the model
add_hook_to_module(model, optimizer_hook)

# 3. auto_manage will now manage the hooked, optimizable model
with auto_manage(model):
    # On the first run, the hook will profile and optimize the model.
    # On subsequent runs, it will execute the optimized plan.
    model(torch.randn(1, 100))
```

### 3.4. Composition with `CommitObjectProxy`

`auto_manage` can also wrap a `CommitObjectProxy`, allowing you to combine state management with resource management.

```python
from sire import CommitObjectProxy, CallableCommit

# Base model is the optimized one from the previous example
proxy = CommitObjectProxy(model)

def my_patch(m):
    # A function that patches the model
    print("Applying patch!")
    # ... patch logic ...
    def revert(m):
        print("Reverting patch!")
        # ... revert logic ...
    return revert

commit = CallableCommit(my_patch)
patched_proxy = proxy.clone_and_add_commit(commit)

# Manage the proxy that represents the patched state
with auto_manage(patched_proxy):
    # The proxy will apply the patch, and the resource manager will
    # ensure the underlying (optimized) model is on the GPU.
    patched_proxy.get_current_object()(torch.randn(1, 100))
```
This demonstrates how `CommitObjectProxyWrapper` seamlessly integrates with `TorchModuleWrapper` and `InferenceOptimizerHook`, allowing all components to be composed flexibly.
