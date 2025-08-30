# Developer Guide: ECS Digital Asset Management (DAM) System

## 1. Introduction

This document provides guidance for developers working on the ECS Digital Asset Management (DAM) system. This project implements a DAM using an Entity-Component-System (ECS) architecture in Python. The core idea is to represent assets (images, videos, audio, documents, etc.) as Entities (simple IDs), which are then described by attaching various Components (data-only dataclasses). Systems (typically functions or dedicated modules) operate on entities based on the components they possess.

## 2. Core Architectural Concepts

The system is built upon the Entity-Component-System (ECS) pattern, which promotes flexibility and modularity.

### 2.1. Entities
-   **Definition**: Entities are unique identifiers (typically integers or UUIDs) representing a single digital asset or concept within the system. They don't hold data themselves but act as a central point to which Components are attached.
-   **Implementation**: In our system, Entities are represented by the `dam.models.entity.Entity` SQLAlchemy model, which primarily provides a unique `id`.

### 2.2. Components
-   **Definition**: Components are data-only objects that describe a specific aspect or property of an entity. Each component type defines a specific piece of data.
-   **Implementation**:
    -   Components inherit from `dam.models.base_component.BaseComponent`.
    - Dataclass behavior is inherited from `dam.models.core.base_class.Base`.
    - Components are located in the various `dam_media_*` packages.

### 2.3. BaseComponent
-   Provides common fields: `id`, `entity_id` (FK to `entities.id`), and an `entity` relationship.

### 2.4. Systems
-   **Definition**: Systems encapsulate the logic that operates on entities. They can be triggered in several ways:
    - By the scheduler to run at a specific `SystemStage`.
    - By listening for a broadcast `Event`.
    - By handling a dispatched `Command`.
-   **Implementation**:
    *   Systems are Python functions decorated with `@system`, `@listens_for`, or `@handles_command`.
    *   They are organized into modules within the `systems/` directory of each package.

### 2.5. Plugins
-   **Definition**: The DAM system is built on a plugin architecture. Each plugin is responsible for registering its own components, systems, and resources.
-   **Implementation**:
    *   Plugins implement the `dam.core.plugin.Plugin` protocol.
    *   The `dam_app` package is responsible for loading plugins.
    *   Plugins can depend on other plugins. The `world.add_plugin()` method prevents duplicate registration.

## 3. Project Structure

A brief overview of the key packages:

*   `dam`: The core framework, providing the ECS building blocks.
*   `dam_app`: The main CLI application, which loads and configures plugins.
*   `dam_media_image`: A plugin for image-related functionality.
*   `dam_media_audio`: A plugin for audio-related functionality.
*   `dam_media_transcode`: A plugin for transcode-related functionality.
*   `dam_psp`: An optional plugin for PSP ISO ingestion.
*   `dam_semantic`: An optional plugin for semantic search.

---

## 4. Guide: Adding a New Component or System

This section walks through the process of adding new functionality to the DAM system.

### 4.1. Guideline for New Systems

When adding a new system, first consider if it can be added to an existing plugin package (e.g., `dam_media_image`, `dam_psp`). If the new system provides functionality that is closely related to an existing plugin, it should be added to that plugin.

If the new system is not a good fit for an existing plugin, create a new plugin package for it. This keeps the codebase modular and allows for optional loading of functionality.

### 4.2. Adding a New Component

The process for adding a new component is as follows:
1.  **Define the Component:** Create a new component class in the appropriate plugin package (e.g., `dam_media_image/models/`).
2.  **Register the Component:** Ensure the component is imported in the `__init__.py` of its package so that SQLAlchemy is aware of it.
3.  **Create a System:** Create a system to operate on the new component.
4.  **Register the System:** Register the system in the plugin's `build` method.

### 4.3. Adding a New Command and Handler

The Command pattern is used for imperative actions where the caller requests a specific operation to be performed.

1.  **Define the Command:**
    - In the appropriate package, create a `commands.py` file if it doesn't exist.
    - Define a new dataclass that inherits from `dam.core.commands.BaseCommand`.
    - Add fields to the dataclass to carry the necessary data for the operation.

    *Example (`packages/my_plugin/commands.py`):*
    ```python
    from dataclasses import dataclass
    from dam.core.commands import BaseCommand

    @dataclass
    class RenameAssetCommand(BaseCommand):
        entity_id: int
        new_name: str
    ```

2.  **Create the Command Handler System:**
    - In the package's `systems/` module, create a new function to handle the command.
    - Decorate the function with `@handles_command(YourCommandClass)`.
    - The function must be `async` and its first argument should be the command object.
    - Use functions to perform the business logic.

    *Example (`packages/my_plugin/systems/asset_systems.py`):*
    ```python
    from dam.core.systems import handles_command
    from dam.core.transaction import EcsTransaction
    from my_plugin.commands import RenameAssetCommand
    from my_plugin.models import NameComponent # Assuming a component that stores the name

    @handles_command(RenameAssetCommand)
    async def handle_rename_asset_command(
        cmd: RenameAssetCommand,
        transaction: EcsTransaction,
    ):
        print(f"Handling command to rename entity {cmd.entity_id} to '{cmd.new_name}'")

        # Use the transaction object to interact with the database
        name_component = await transaction.get_component(cmd.entity_id, NameComponent)

        if name_component:
            name_component.name = cmd.new_name
            # The transaction object automatically handles registering the change
            # because the component is still managed by the underlying session.
            print(f"Name for entity {cmd.entity_id} changed in transaction.")
        else:
            # Or create a new component
            new_name_component = NameComponent(name=cmd.new_name)
            await transaction.add_component_to_entity(cmd.entity_id, new_name_component)
            print(f"New name for entity {cmd.entity_id} added in transaction.")

    ```

3.  **Register the Handler:**
    - In your plugin's `build` method, register the system and associate it with the command.

    *Example (`packages/my_plugin/plugin.py`):*
    ```python
    from .commands import RenameAssetCommand
    from .systems.asset_systems import handle_rename_asset_command

    class MyPlugin(Plugin):
        def build(self, world: "World") -> None:
            world.register_system(
                handle_rename_asset_command,
                command_type=RenameAssetCommand,
            )
    ```

4.  **Dispatch the Command:**
    - From anywhere in the application that has access to a `World` object, you can dispatch the command.

    *Example:*
    ```python
    from my_plugin.commands import RenameAssetCommand

    # ... get world object ...
    command = RenameAssetCommand(entity_id=123, new_name="My Cool Asset")
    await world.dispatch_command(command)
    ```

---

## 5. Other Development Aspects

### 5.1. Database Migrations (Alembic Workflow)
-   **Current Status (Important):** Alembic is set up, but its usage for generating and applying migrations is **currently paused**.
-   **Development Database Setup:** For development, use the `dam-cli setup-db` command.

### 5.2. Running Tests

The project uses `pytest` for testing, preferably run via `uv` and `poe`.
-   **Run all tests**:
    ```bash
    uv run poe test
    ```
-   **Test Coverage**:
    ```bash
    uv run poe test-cov
    ```

### 5.3. Code Style and Conventions

-   **Formatting & Linting**: `uv run poe format` and `uv run poe lint`.
-   **Type Checking**: `uv run poe mypy`.
