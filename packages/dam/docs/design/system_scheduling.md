# Design Document: Advanced System Scheduling

## 1. Introduction

This document outlines a comprehensive redesign of the system execution and scheduling mechanism in the `dam` framework. The goal is to move beyond simple parallel execution to a sophisticated, resource-aware, and cost-based model that maximizes throughput and efficiency.

This design addresses three core areas:
1.  **Hierarchical Resource Management**: A centralized system for defining, tracking, and limiting concurrent access to specific, hierarchical resources (e.g., I/O on a particular disk, memory on a specific GPU).
2.  **Resource-Aware Execution**: An enhanced `SystemExecutor` that schedules systems based on resource availability.
3.  **Pluggable, Cost-Based Command Dispatch**: An intelligent, extensible command dispatcher that selects the most efficient system to handle a command based on dynamic, plugin-provided "cost" calculations.

## 2. Hierarchical Resource Management

### 2.1. The `ResourceManager` and `ResourcePool`

The core idea is to move from simple named resources to a hierarchical model of `ResourcePools`. A pool represents a quantity of a resource that can be acquired and released. Pools can contain sub-pools, allowing for fine-grained control.

```python
# In dam.core.resources
class ResourcePool:
    def __init__(self, name: str, limit: int):
        self.name = name
        self.limit = limit
        self.current_usage = 0
        self.sub_pools: dict[str, ResourcePool] = {}
        self._lock = asyncio.Semaphore(limit)

    async def acquire(self):
        await self._lock.acquire()

    def release(self):
        self._lock.release()

class ResourceManager:
    def __init__(self):
        self._root_pools: dict[str, ResourcePool] = {}

    def get_pool(self, path: str) -> ResourcePool | None:
        # Path like "disk/read/sda1" or "gpu/0/memory"
        parts = path.split('/')
        pool = self._root_pools.get(parts[0])
        for part in parts[1:]:
            if not pool: return None
            pool = pool.sub_pools.get(part)
        return pool
```

This structure allows us to model resources with high specificity. For example:
*   `disk/read/sda`: A pool for read operations on the `sda` disk.
*   `gpu/0/memory`: A pool for memory usage on GPU `0`.
*   `memory/system`: A general pool for system RAM.

### 2.2. Resource Declaration and Configuration

Resource pools will be configurable in `dam.toml` using a nested structure.

```toml
[world.resources.disk.read]
# Default limit for all disk reads
default_limit = 4

[world.resources.disk.read.sda]
limit = 1 # sda is a slow HDD

[world.resources.disk.read.nvme0]
limit = 8 # nvme0 is a fast SSD

[world.resources.gpu.0]
memory = { limit = 8192 } # 8GB
cores = { limit = 1024 }
```

Auto-detection and dynamic runtime modification will also be supported through commands that can target specific pool paths.

## 3. Resource-Aware Execution

### 3.1. Declaring Resource Requirements

The `@system` decorator will be extended to accept specific resource paths.

```python
from dam.core.systems import system

@system(
    on_command=ProcessImageCommand,
    resources={"gpu/0/memory": 1024, "disk/read/nvme0": 1}
)
async def process_image_on_fast_disk(cmd: ProcessImageCommand):
    # This system will only be scheduled if it can acquire 1024MB of memory
    # on GPU 0 AND one read slot on the nvme0 disk.
```

The `SystemExecutor` will be updated to acquire and release locks on the specific `ResourcePool` semaphores, ensuring that resource limits are respected across the hierarchy.

## 4. Pluggable, Cost-Based Command Dispatch

To support user-defined logic, the cost calculation and allocation strategy will be made pluggable.

### 4.1. The `Cost` and `AllocationStrategy` Protocols

The `Cost` dataclass remains, but we now introduce an `AllocationStrategy` protocol.

```python
# in dam.core.costs
from typing import Protocol

@dataclass(order=True)
class Cost:
    """Represents the cost of executing a system."""
    priority: int = 100 # Lower is better
    time_ms: int = 0
    io_read_bytes: int = 0

class AllocationStrategy(Protocol):
    """A protocol for a strategy that selects the best system."""
    async def calculate_cost(
        self,
        system: Callable[..., Any],
        command: BaseCommand[Any, Any],
        world: World
    ) -> Cost:
        ...

    async def select_system(
        self,
        systems_with_costs: list[tuple[Callable[..., Any], Cost]]
    ) -> Callable[..., Any] | None:
        # Default strategy: return the one with the lowest cost.
        ...
```

### 4.2. Plugin-Defined Strategies

Plugins can define their own allocation strategies and register them with the `World`. This allows for highly specialized logic. For example, a plugin could create a strategy that considers network latency when choosing between a local file and a cloud-stored file.

A system can specify which allocation strategy should be used to evaluate it, or a default strategy can be used.

```python
# In a plugin's system file
class NetworkAwareStrategy(AllocationStrategy):
    async def calculate_cost(self, system, command, world) -> Cost:
        # Custom logic to ping a server or check bandwidth
        ...

@system(
    on_command=GetAssetStreamCommand,
    strategy=NetworkAwareStrategy()
)
class GetStreamFromCloud:
    ...
```

### 4.3. Modified Command Dispatcher

The `WorldScheduler.dispatch_command` method will be updated:

1.  Find all systems registered for the command.
2.  For each system, determine its `AllocationStrategy` (either from the decorator or a world default).
3.  Use the appropriate strategy to calculate the cost for each system.
4.  If the command's `ExecutionStrategy` is `BEST_ONE`, use the strategy's `select_system` method to choose the handler. This allows a strategy to implement complex selection logic beyond just picking the minimum cost (e.g., load balancing).
5.  If the strategy is `PARALLEL`, costs can be used to prioritize execution.

This deeply extensible design provides a powerful framework for intelligent system scheduling, giving developers precise control over resource management and execution logic, ensuring the DAM operates at maximum efficiency.
