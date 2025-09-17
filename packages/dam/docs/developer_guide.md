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
-   **Implementation**: All components inherit from a common `dam.models.core.base_component.Component` base class. This ensures they can be discovered by the framework. There are two main types of components:
    -   **Standard Components**: Inherit from `dam.models.core.base_component.BaseComponent`. An entity can have **multiple** instances of these components. They have their own `id` primary key.
    -   **Unique Components**: Inherit from `dam.models.core.base_component.UniqueComponent`. An entity can only have **one** instance of these components. They use the `entity_id` as their primary key to enforce this uniqueness at the database level.
    - Dataclass behavior is inherited from `dam.models.core.base_class.Base`.
    - Components are located in the various `dam_media_*` packages.

### 2.3. Component Base Classes
-   **`Component`**: The abstract root of all components.
-   **`BaseComponent`**: Provides common fields for standard (non-unique) components: an auto-incrementing `id` primary key, and the `entity_id` foreign key.
-   **`UniqueComponent`**: Provides the `entity_id` as a primary key for unique components.

### 2.4. Systems
-   **Definition**: Systems encapsulate the logic that operates on entities. They can be triggered in several ways:
    - By the scheduler to run at a specific `SystemStage`.
    - By listening for a broadcast `Event`.
    - By handling a dispatched `Command`.
-   **Implementation**:
    *   Systems are Python functions decorated with `@system`.
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

### 4.1. Architectural Preference: Commands over Events/Stages

A key design principle for the `dam` ecosystem is to **prefer the Command pattern for implementing new functionality**.

-   **Why?**: Commands provide a clear, imperative, and traceable control flow. When you dispatch a command, you have a clear expectation of a specific action being performed. This makes the system easier to understand, debug, and test.
-   **Guideline**: Unless a task's requirements explicitly call for a decoupled, event-driven workflow (e.g., multiple independent systems reacting to a single occurrence) or a lifecycle-based stage, you should implement the logic as a command and its corresponding handler system. Avoid using events or component markers as the primary mechanism for triggering core business logic.

### 4.2. Guideline for New Systems

When adding a new system, first consider if it can be added to an existing plugin package (e.g., `dam_media_image`, `dam_psp`). If the new system provides functionality that is closely related to an existing plugin, it should be added to that plugin.

If the new system is not a good fit for an existing plugin, create a new plugin package for it. This keeps the codebase modular and allows for optional loading of functionality.

### 4.3. Adding a New Component

The process for adding a new component is as follows:
1.  **Define the Component:** Create a new component class in the appropriate plugin package (e.g., `dam_media_image/models/`).
2.  **Register the Component:** Ensure the component is imported in the `__init__.py` of its package so that SQLAlchemy is aware of it.
3.  **Create a System:** Create a system to operate on the new component.
4.  **Register the System:** Register the system in the plugin's `build` method.

### 4.4. Adding a New Command and Handler

The Command pattern is used for imperative actions where the caller requests a specific operation to be performed.

1.  **Define the Command:**
    - In the appropriate package, create a `commands.py` file if it doesn't exist.
    - Define a new dataclass that inherits from an appropriate base command in `dam.core.commands`. For tasks that analyze a single entity, `AnalysisCommand` is a good choice as it provides common fields and helpers.
    - Add any additional fields the command needs.

    *Example (`packages/my_plugin/commands.py`):*
    ```python
    from dataclasses import dataclass
    from dam.core.commands import AnalysisCommand

    @dataclass
    class ExtractDominantColorCommand(AnalysisCommand[None]):
        """A command to extract the dominant color from an image asset."""
        # entity_id, depth, and stream are inherited from AnalysisCommand
        pass # No extra fields needed for this simple command
    ```

2.  **Create the Command Handler System:**
    - In the package's `systems/` module, create a new function to handle the command.
    - Decorate it with `@system(on_command=YourCommandClass)`.
    - Use the command's helper methods and the `EcsTransaction` object to implement the logic.

    *Example (`packages/my_plugin/systems/asset_systems.py`):*
    ```python
    from dam.core.systems import system
    from dam.core.transaction import EcsTransaction
    from dam.core.world import World
    from my_plugin.commands import ExtractDominantColorCommand
    from my_plugin.models import DominantColorComponent # A UniqueComponent

    @system(on_command=ExtractDominantColorCommand)
    async def handle_extract_dominant_color_command(
        cmd: ExtractDominantColorCommand,
        transaction: EcsTransaction,
        world: World, # Inject the world to use command helpers
    ):
        print(f"Handling command to extract color from entity {cmd.entity_id}")

        # Use the command's helper to get a file stream
        try:
            image_stream = await cmd.get_stream(world)
        except ValueError as e:
            print(f"Error: {e}")
            return

        # ... process the stream to find the dominant color (e.g., '#FF0000') ...
        dominant_color_hex = "#FF0000" # Placeholder

        # Create the component instance
        color_component = DominantColorComponent(hex_color=dominant_color_hex)

        # Use the transaction's helper to add or update the component.
        # Since DominantColorComponent is a UniqueComponent, this will correctly
        # add it if it's new, or update it if it already exists.
        await transaction.add_or_update_component(cmd.entity_id, color_component)
        print(f"Dominant color for entity {cmd.entity_id} set to {dominant_color_hex}")
    ```

3.  **Dispatch the Command:**
    - From anywhere in the application that has access to a `World` object, you can dispatch the command.
    - The `dispatch_command` method returns a `SystemExecutor` object, which you can use to get the results.

    *Example:*
    ```python
    from my_plugin.commands import ExtractDominantColorCommand

    # ... get world object ...
    command = ExtractDominantColorCommand(entity_id=123)

    # Dispatch the command and get the executor
    executor = world.dispatch_command(command)

    # You can "fire and forget" by simply awaiting the executor to run it
    # (though this is less common for commands that return values)
    # async for _ in executor:
    #     pass

    # Or, more commonly, use one of the helper methods to get the results.
    # For example, if the handler for RenameAssetCommand returns the new name:
    # new_name_result = await executor.get_one_value()
    # print(f"Command returned: {new_name_result}")
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

### 5.2.1. Test Fixtures

The project uses a shared fixture model to simplify testing across the `dam` core and its plugins. The core test fixtures are located in the `packages/dam_test_utils` package.

#### Using Shared Fixtures

To use the shared fixtures in a plugin's test suite, add the following line to your package's `tests/conftest.py` file:

```python
pytest_plugins = ["dam_test_utils.fixtures"]
```

This will make all the fixtures defined in `dam_test_utils` available to your tests. This includes fixtures for setting up a test database, creating a `World` instance, and generating sample data.

#### Creating Package-Specific Fixtures

If your package requires specific fixtures that are not provided by `dam_test_utils`, you can define them in your package's `tests/conftest.py` file alongside the `pytest_plugins` line. These fixtures will be available to all tests within your package.

### 5.3. Code Style and Conventions

-   **Formatting & Linting**: `uv run poe format` and `uv run poe lint`.
-   **Type Checking**: `uv run poe mypy`.

### 5.4. Testing CLI Commands

When writing tests for `click` or `typer` commands, **do not use `click.testing.CliRunner` or `typer.testing.CliRunner`**. Instead, directly test the functions that the commands call. This ensures that the business logic is tested independently of the command-line interface.

**Incorrect:**

```python
from click.testing import CliRunner
from dam.cli import hello

def test_hello():
    runner = CliRunner()
    result = runner.invoke(hello, ["Jules"])
    assert result.exit_code == 0
    assert "Hello, Jules!" in result.output
```

**Correct:**

Refactor the code to separate the business logic:

```python
# In your application code
def get_greeting(name):
    return f"Hello, {name}!"

@click.command()
@click.argument("name")
def hello(name):
    """Says hello to a user."""
    print(get_greeting(name))
```

And then test the business logic directly:

```python
# In your test code
from dam.core import get_greeting

def test_get_greeting():
    assert get_greeting("Jules") == "Hello, Jules!"
```
