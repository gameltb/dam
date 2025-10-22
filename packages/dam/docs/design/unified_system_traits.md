# Design Document: The Unified Trait System

## 1. Introduction and Vision

This document presents a unified and cohesive design for managing system capabilities and execution characteristics within the `dam` framework. It refactors the previously separate concepts of "Asset Traits" and "Resource Scheduling" into a single, powerful abstraction: the **Trait**.

The core vision is that a **Trait** is a universal, context-dependent descriptor. This single concept can be used to describe:
1.  **The capabilities of assets**: What operations can be performed on an entity, based on its components (e.g., it is `AssetContentReadable`).
2.  **The characteristics of systems**: How a system behaves during execution (e.g., its resource usage, its cost, whether it is I/O-bound).

Furthermore, this design introduces a crucial feedback loop: an **Execution Observer** that allows the framework to learn a system's actual performance traits over time and adapt its scheduling decisions for future tasks.

## 2. Core Concept: Traits as Universal Descriptors

A Trait is a data-carrying object that represents a specific characteristic or capability.

### 2.1. Component Traits: Describing "What"

A Component Trait describes a capability that an entity possesses due to its components.

*   **Definition**: A trait is defined as a class, often containing abstract command definitions.
    *   *Example:* The `AssetContentReadable` trait signifies that an entity's content can be read as a byte stream.
*   **Implementation and Binding**: A plugin registers an implementation for a trait, binding it to a specific component class or a **tuple of component classes**.
    *   If a trait is bound to a single component (e.g., `FileLocationComponent`), any entity with that component has the trait.
    *   If a trait is bound to a tuple (e.g., `(ZipArchiveComponent, ZipEntryComponent)`), the trait only applies to entities that possess **all** components in that tuple.
*   **Discovery**: The `TraitManager` and `world.get_available_traits_for_entity` are the foundation for discovering which traits an entity has at runtime based on these rules.

### 2.2. System Traits: Describing "How"

A System Trait describes how a system behaves when it executes. These traits are the key input for the scheduler.

*   **Definition**: A System Trait is a dataclass representing a specific execution characteristic.
*   **Examples**:
    ```python
    @dataclass
    class ResourceUsageTrait:
        """Declares that a system requires a specific resource."""
        pool_path: str  # e.g., "disk/read/nvme0"
        amount: int = 1

    @dataclass
    class ExecutionCostTrait:
        """The predicted cost of execution."""
        priority: int = 100 # Lower is better
        time_ms: int = 0

    @dataclass
    class IOBoundTrait:
        """A marker trait indicating the system spends most of its time waiting for I/O."""
    ```

### 2.3. Context-Dependent Trait Calculation

A system's traits are not static. Systems must provide a method to calculate their traits dynamically based on the command context.

```python
# In a system file
@system(on_command=GetAssetStreamCommand)
class GetStreamFromZipArchive:
    async def __call__(self, cmd: GetAssetStreamCommand, world: World):
        ...

    @classmethod
    async def calculate_traits(
        cls, cmd: GetAssetStreamCommand, world: World
    ) -> list[Trait]:
        """Dynamically calculate the traits for this execution."""
        # This implementation requires reading from a potentially large zip file.
        traits = [
            ResourceUsageTrait(pool_path="disk/read/hdd0", amount=1),
            ExecutionCostTrait(priority=50, time_ms=100),
            IOBoundTrait(),
        ]
        return traits
```

## 3. Practical Application: Implementing Asset Capabilities

To make the concept of Component Traits concrete, this section provides a full, step-by-step example of how the `AssetContentReadable` trait would be implemented for assets represented by a `FileLocationComponent`.

### Step 1: Define the Trait and its Commands

First, we define the `AssetContentReadable` trait. This class acts as a namespace for its abstract commands.

```python
# in dam.traits.asset_content
from dam.commands.core import EntityCommand

class AssetContentReadable(Trait):
    """A trait for components that represent asset content that can be read as a stream of bytes."""
    name = "asset.content.readable"
    description = "Provides a way to read the raw content of an asset."

    @dataclass
    class GetStream(EntityCommand[StreamProvider, None]):
        """Abstract command to get the content as a stream."""

    @dataclass
    class GetSize(EntityCommand[int, None]):
        """Abstract command to get the content size in bytes."""
```

### Step 2: Implement the Concrete System Logic

Next, a plugin (e.g., `dam-fs`) provides the concrete system(s) that handle the logic for a specific component. This system is a standard command handler.

```python
# in packages/dam-fs/src/dam_fs/systems.py
import os
import aiofiles
from dam.core.systems import system
from dam.core.traits.asset_content import AssetContentReadable
from .components import FileLocationComponent # The component this system works with

@system(on_command=AssetContentReadable.GetStream)
async def get_stream_from_file(
    cmd: AssetContentReadable.GetStream,
    world: World
) -> StreamProvider | None:
    """Handles the GetStream command for entities with a FileLocationComponent."""
    file_loc = await world.db.get_component(cmd.entity_id, FileLocationComponent)
    if not file_loc or not os.path.exists(file_loc.path):
        return None

    return FileStreamProvider(file_loc.path)

@system(on_command=AssetContentReadable.GetSize)
async def get_size_from_file(
    cmd: AssetContentReadable.GetSize,
    world: World
) -> int:
    """Handles the GetSize command for entities with a FileLocationComponent."""
    file_loc = await world.db.get_component(cmd.entity_id, FileLocationComponent)
    if not file_loc or not os.path.exists(file_loc.path):
        return 0
    return os.path.getsize(file_loc.path)
```

### Step 3: Register the Trait Implementation

The plugin's `build` method connects the trait, the component, and the system implementation.

```python
# in packages/dam-fs/src/dam_fs/plugin.py
from dam.core.plugin import Plugin
from dam.core.world import World
from dam.traits import TraitImplementation
from dam.traits.asset_content import AssetContentReadable
from .components import FileLocationComponent
from .systems import get_stream_from_file, get_size_from_file

class FileSystemPlugin(Plugin):
    def build(self, world: World):
        readable_impl = TraitImplementation(
            trait=AssetContentReadable,
            handlers={
                AssetContentReadable.GetStream: get_stream_from_file,
                AssetContentReadable.GetSize: get_size_from_file,
            }
        )
        world.trait_manager.register(FileLocationComponent, readable_impl)
```

### Step 4: Discover and Use the Trait

Finally, a consumer can discover and use this trait without needing to know about `FileLocationComponent`.

```python
async def print_asset_size(entity_id: int, world: World):
    available_traits = await world.get_available_traits_for_entity(entity_id)
    readable_trait = next((t for t in available_traits if isinstance(t, AssetContentReadable)), None)

    if readable_trait:
        executor = world.dispatch_command(AssetContentReadable.GetSize(entity_id=entity_id))
        size = await executor.get_one_value()
        print(f"Asset size is: {size} bytes")
    else:
        print(f"Entity {entity_id} does not support reading content.")
```

## 4. The Scheduler and Observer: A Learning System

This system creates a feedback loop to enable adaptive performance, using an `ExecutionObserver` to listen to performance metrics from the `SystemExecutor` and storing them in a `SystemPerformanceProfile`.

## 5. The Unified Scheduling Workflow

The scheduling process integrates these concepts:
1.  **Command Dispatch**: Find potential handlers.
2.  **Trait Aggregation**: Gather predicted traits (from `calculate_traits`) and observed traits (from `SystemPerformanceProfile`).
3.  **Intelligent Selection**: A pluggable `AllocationStrategy` uses the traits to make an informed decision.
4.  **Resource-Aware Execution**: The `SystemExecutor` uses `ResourceUsageTrait`s to acquire resource locks.
5.  **Observation and Learning**: The executor publishes an `ExecutionRecord`, updating the profile.

## 6. Advanced Features and Operational Considerations

### 6.1. Scheduler Observability & Debugging

A `debug_dispatch` method will be added to the `WorldScheduler`, which will return a structured `SchedulingReport` object. This report will provide a clear audit trail of the scheduler's decision-making process.

### 6.2. Explicit System Targeting (The Escape Hatch)

The `dispatch_command` method will accept an optional `target_system` parameter to bypass the dynamic selection logic and guarantee predictable execution when required.

### 6.3. Automated Trait Generation via Execution Tracing

This is a forward-looking extension to automate the creation of accurate `System Traits`.
*   **Phase 1: Deep Execution Tracing**: A `SystemTracer` component will use low-level technologies (e.g., eBPF) to capture high-fidelity data.
*   **Phase 2: Diagnostic Reporting**: The trace data is processed into a structured `DiagnosticReport`.
*   **Phase 3: AI-Assisted Trait Generation**: The `DiagnosticReport` is used to prompt an LLM to generate the `calculate_traits` method code, turning a manual task into an AI-assisted review process.

## 7. Conclusion

This unified design provides a powerful and elegant architecture. By abstracting capabilities and characteristics into the single concept of a "Trait," and by incorporating advanced operational tools like observability, escape hatches, and a roadmap for AI-assisted development, this framework is poised to be highly efficient, resilient, and adaptable to future challenges.
