# Design Document: Trait-Based Asset Operations

## 1. Introduction

This document proposes a redesign of the `AssetOperation` concept into a more formal, discoverable, and extensible "Trait" system. The current `AssetOperation` class is a simple dataclass that couples a name and description to a set of command classes. This design is limiting and doesn't provide a robust way to discover capabilities or handle different implementations for the same conceptual operation.

The new Trait system will provide:
1.  **Formal Trait Definitions**: Traits as explicit declarations of a capability, defined by a set of abstract commands.
2.  **Decoupled Implementations**: Components can "implement" traits, providing concrete handlers for the abstract commands. A trait can be implemented for a single component or a combination of components.
3.  **Discovery and Registry**: A central registry to discover which traits an entity or component set supports, find implementations, and prevent conflicts.

## 2. The Trait System

### 2.1. Trait Definition

A Trait is a class that defines a capability. It specifies a set of abstract commands that must be implemented by any component (or set of components) that claims to support this trait. Trait names should be specific to their domain.

```python
# In dam.core.traits
from typing import Protocol

class Trait(Protocol):
    """Base protocol for all traits."""
    name: str
    description: str

class AssetContentReadable(Trait):
    """A trait for components that represent asset content that can be read as a stream of bytes."""
    name = "asset.content.readable"
    description = "Provides a way to read the raw content of an asset."

    class GetStreamCommand(EntityCommand[AsyncIterator[bytes], None]): ...
    class GetSizeCommand(EntityCommand[int, None]): ...
```

Here, `AssetContentReadable` is a trait that defines two abstract commands. Any component combination that wants to be "readable" must provide a way to handle these two commands.

### 2.2. Trait Implementation

A trait is implemented by providing concrete systems that handle the trait's abstract commands for a specific component class or a tuple of component classes. This is done via a registration process.

**Example 1: Single-Component Trait (File System)**
```python
# In some plugin's systems file

@system(on_command=AssetContentReadable.GetStreamCommand)
async def get_stream_from_file(cmd: AssetContentReadable.GetStreamCommand, world: World):
    # Logic to get a FileLocationComponent from cmd.entity_id
    # and yield a file stream.
    ...
```

**Example 2: Multi-Component Trait (Archiving)**
Some traits only make sense in the context of multiple components. For example, an `Archivable` trait might need a file path *and* some metadata.

```python
# Trait definition
class Archivable(Trait):
    name = "asset.archivable"
    description = "Provides a way to archive an asset to a target location."
    class ArchiveCommand(EntityCommand[None, None]):
        target: str

# System implementation that requires two components
@system(on_command=Archivable.ArchiveCommand)
async def archive_asset(cmd: Archivable.ArchiveCommand, world: World):
    # This system would use DI to query for both components:
    file_loc = world.db.get_component(cmd.entity_id, FileLocationComponent)
    metadata = world.db.get_component(cmd.entity_id, AssetMetadataComponent)
    if not file_loc or not metadata:
        # Cannot handle this command
        return
    # ... logic to archive the file using its path and metadata ...
```

### 2.3. The `TraitManager` and Registration

The `TraitManager` will handle the registration and discovery of trait implementations. It will support registration against a single component type or a tuple of types.

```python
# In dam.core.traits
from typing import Type, Union, Tuple

ComponentSelector = Union[Type, Tuple[Type, ...]]

class TraitManager:
    def __init__(self):
        # (frozenset({component_type, ...}), trait_type) -> implementation
        self._registry: dict[tuple[frozenset[type], type], TraitImplementation] = {}

    def _normalize_selector(self, selector: ComponentSelector) -> frozenset[type]:
        if isinstance(selector, tuple):
            return frozenset(selector)
        return frozenset([selector])

    def register(self, selector: ComponentSelector, trait: Trait, implementation: TraitImplementation):
        key = (self._normalize_selector(selector), type(trait))
        ...

    def get_implementation(self, selector: ComponentSelector, trait_type: type) -> TraitImplementation | None:
        key = (self._normalize_selector(selector), trait_type)
        ...

    def get_traits_for_selector(self, selector: ComponentSelector) -> list[Trait]:
        ...
```

The registration in a plugin's `build` method is now more flexible:
```python
# In a plugin's plugin.py
class MyFileSystemPlugin(Plugin):
    def build(self, world: World):
        # Single-component trait
        readable_impl = TraitImplementation(...)
        world.trait_manager.register(FileLocationComponent, AssetContentReadable, readable_impl)

        # Multi-component trait
        archivable_impl = TraitImplementation(...)
        world.trait_manager.register(
            (FileLocationComponent, AssetMetadataComponent),
            Archivable,
            archivable_impl
        )
```

## 3. Using the Trait System

### 3.1. Discovering Traits for an Entity

A key requirement is to determine which traits are available for a given entity at runtime. This can be accomplished with a helper method on the `World`.

```python
# In dam.core.world.World
from itertools import combinations

class World:
    # ... existing methods ...
    async def get_available_traits_for_entity(self, entity_id: int) -> list[Trait]:
        """
        Inspects an entity's components and returns a list of all traits
        that can be fulfilled by them.
        """
        components = await self.db.get_components_for_entity(entity_id)
        component_types = {type(c) for c in components}

        available_traits = set()

        # Check all possible combinations of the entity's components
        for i in range(1, len(component_types) + 1):
            for selector_tuple in combinations(component_types, i):
                traits = self.trait_manager.get_traits_for_selector(selector_tuple)
                available_traits.update(traits)

        return list(available_traits)
```

### 3.2. Dispatching Trait Commands

With a way to discover traits, a high-level function can operate on an entity without knowing its underlying components.

```python
async def read_asset_content(entity_id: int, world: World):
    available_traits = await world.get_available_traits_for_entity(entity_id)

    if AssetContentReadable not in available_traits:
        return None

    # We know the trait is available, now find the implementation.
    # (This logic can be encapsulated in a helper function)
    components = await world.db.get_components_for_entity(entity_id)
    component_types = {type(c) for c in components}

    for i in range(1, len(component_types) + 1):
        for selector_tuple in combinations(component_types, i):
            impl = world.trait_manager.get_implementation(selector_tuple, AssetContentReadable)
            if impl:
                handler = impl.handlers[AssetContentReadable.GetStreamCommand]
                executor = world.scheduler.execute_one_time_system(
                    handler,
                    command_object=AssetContentReadable.GetStreamCommand(entity_id=entity_id)
                )
                return await executor.get_one_value()
    return None
```

## 4. Advantages of this Design

*   **Decoupling**: Logic can depend on abstract capabilities (Traits) rather than concrete component types.
*   **Discoverability**: The system can dynamically determine an entity's capabilities at runtime, which is ideal for UIs and plugins.
*   **Composability**: Traits can be defined for combinations of components, allowing for more complex and specific capabilities to be modeled.
*   **Extensibility**: New components and plugins can provide implementations for existing traits without modifying any core code.
*   **Clarity and Safety**: The registration process ensures that the link between a component set, a trait, and its implementation is explicit and unambiguous.
