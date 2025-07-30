# Sire - Design Specification

## 1. Overview

Sire is a dynamic, just-in-time resource manager for PyTorch models. Its primary goal is to enable the efficient execution of multiple models on a single machine with constrained hardware, particularly GPU memory. It achieves this by abstracting memory management and dynamically moving models between a high-speed "runtime" device (like a GPU) and a larger "offload" storage area (like CPU RAM).

The design is modular and built on a few core concepts that work together to provide automatic resource management.

## 2. Core Concepts

### 2.1. Resource Pools (`ResourcePool`)

A `ResourcePool` represents a source of memory on a specific device.
- **Implementations**: `ResourcePoolCUDA` and `ResourcePoolCPU` are the default implementations.
- **Function**: Each pool is responsible for tracking its own available memory. For example, `ResourcePoolCUDA` checks the free VRAM on its assigned GPU.
- **Management**: A central singleton, `ResourcePoolManagement`, keeps a registry of all available resource pools, keyed by the `torch.device` object.

### 2.2. Resource Users (`ResourcePoolUserABC`)

A "Resource User" is any object that consumes memory from a `ResourcePool`.
- **Abstraction**: The `ResourcePoolUserABC` (Abstract Base Class) defines the interface for these objects.
- **Implementation**: `TorchModuleWrapper` is the primary implementation, designed to wrap a `torch.nn.Module`.
- **Responsibilities**: A Resource User must be able to:
    - Report how much memory it is currently using on a given device (`get_used_resource_size`).
    - Define its "runtime" and "offload" pools (`on_setup`).
    - Handle being loaded to the runtime device (`on_load`).
    - Handle being evicted from the runtime device to free up memory (`on_resource_request`).

### 2.3. Automatic Management (`AutoManageWrapper`, `auto_manage`)

This is the mechanism that automates the just-in-time loading and offloading.
- **`AutoManageWrapper`**: A generic wrapper that turns any object into a Resource User. It has a type map to select the correct user implementation (e.g., it maps `torch.nn.Module` to `TorchModuleWrapper`). When an object is wrapped, it is registered with the central `ResourcePoolManagement`.
- **`auto_manage` Context Manager**: This is the primary user-facing tool.
    1. **`__enter__`**: When the `with` block is entered, the wrapper for the target object is "locked". It then requests the memory it needs from its `runtime_resource_pool`. The pool, if it doesn't have enough free memory, will ask other (unlocked) Resource Users to offload until the required memory is freed. Finally, the target object is moved to the runtime device (e.g., `model.to(gpu)`).
    2. **`__exit__`**: When the `with` block is exited, the wrapper is "unlocked", making it eligible for eviction if another object needs memory. The `TorchModuleWrapper` is designed to immediately offload its model back to the CPU upon being unlocked.

### 2.4. Hooks (`AutoManageHook`)

For convenience, the automatic management logic can be injected directly into a `torch.nn.Module` using an `AutoManageHook`.
- **Functionality**: This hook attaches to the module's `forward` pass.
- **`pre_forward`**: Before the original `forward` method is called, the hook runs the `__enter__` logic from the `auto_manage` context, moving the model to the GPU.
- **`post_forward`**: After the `forward` method completes, it runs the `__exit__` logic, offloading the model back to the CPU.

### 2.5. Commit-based State Management (`CommitObjectProxy`)

This is an advanced, optional feature for managing different variations of a single base object without deep copying.
- **Commit**: A "commit" is an object representing a reversible change, such as applying a LoRA or monkey-patching a function.
- **`CommitObjectProxy`**: This acts as a proxy to a base object (e.g., a model). You can add a stack of commits to the proxy.
- **`rebase`**: Before the object is used, the proxy ensures the base object is in the correct state by applying or reverting commits as needed to match the desired commit stack. This allows for efficiently switching between different model configurations.
