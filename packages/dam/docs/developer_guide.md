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
-   **Definition**: The DAM system is built on a plugin architecture. Each plugin is responsible for registering its own components, systems, and resources. Plugins are discovered dynamically at runtime using Python's standard entry point mechanism.
-   **Implementation**:
    *   Plugins implement the `dam.core.plugin.Plugin` protocol.
    *   The core `dam` package discovers and loads plugins registered under the `"dam.plugins"` entry point group.
    *   Plugins can depend on other plugins. The `world.add_plugin()` method prevents duplicate registration.

### 2.6. Dynamic World Instantiation

The DAM system supports two primary ways of defining and creating `World` instances:

1.  **From `dam.toml` (Static Worlds)**: The `dam-cli` application can instantiate worlds defined in a `dam.toml` configuration file. This is the standard approach for defining persistent, long-running worlds. The CLI handles the loading and parsing of this file on-demand when a world is requested via the `--world` flag.

2.  **From Entity Components (Dynamic Worlds)**: It is also possible to define a world's entire configuration by attaching multiple `ConfigComponent` instances to a single entity in an existing world. This allows for dynamic, data-driven creation of new, isolated `World` instances. A dedicated system can then be used to:
    -   Query an entity for all its attached `ConfigComponent`s.
    -   Pass this list of components to the `dam.core.world_manager.create_world_from_components` factory function.
    -   This factory will load the corresponding plugins, configure the new world with the component data, and register it for use.

This powerful feature enables scenarios where world configurations are stored and managed within the DAM itself, rather than in static configuration files.

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

## 4. Guide: How to Create a New Plugin

This section details the steps to create and register a new plugin.

1.  **Create a New Package**: Start by creating a new Python package for your plugin (e.g., `dam_my_plugin`).

2.  **Define the Plugin Class**:
    -   In your package, create a module (e.g., `plugin.py`) and define a class that inherits from `dam.core.plugin.Plugin`.
    -   Implement the `build(self, world: World)` method to register all your components, systems, and resources with the `world` object.

    *Example (`packages/dam_my_plugin/src/dam_my_plugin/plugin.py`):*
    ```python
    from dam.core.plugin import Plugin
    from dam.core.world import World
    from .systems import my_system  # Import your system

    class MyPlugin(Plugin):
        """A plugin for my new functionality."""

        # All plugins MUST define a Settings and SettingsComponent class.
        # If the plugin has no settings, these can be empty.
        Settings = MyPluginSettingsModel
        SettingsComponent = MyPluginSettingsComponent

        def build(self, world: World) -> None:
            world.register_system(my_system)
            # Register other components, resources, etc.
    ```

3.  **Define Settings and Configuration Component (Mandatory)**:
    -   Every plugin **must** define a `SettingsModel` and a `SettingsComponent` class, even if the plugin has no configurable settings. This is crucial for the system's dynamic world instantiation capabilities.
    -   Create a `settings.py` file in your plugin's source directory.

    *Example (`packages/dam_my_plugin/src/dam_my_plugin/settings.py`):*
    ```python
    from dam.models.config import ConfigComponent, SettingsModel

    class MyPluginSettingsModel(SettingsModel):
        """Pydantic model for MyPlugin. Can be empty if no settings."""
        pass

    class MyPluginSettingsComponent(ConfigComponent):
        """ECS component for MyPlugin settings. Can be empty."""
        __tablename__ = "dam_my_plugin_config" # Must be unique
    ```

4.  **Register the Plugin via Entry Point**:
    -   In your plugin's `pyproject.toml` file, add a new section for the `dam.plugins` entry point.
    -   Create a key for your plugin (this is the name it will be known by) and set the value to the import path of your plugin class.

    *Example (`packages/dam_my_plugin/pyproject.toml`):*
    ```toml
    [project.entry-points."dam.plugins"]
    my-plugin-name = "dam_my_plugin.plugin:MyPlugin"
    ```
    With this configuration, the DAM application will automatically discover and load `MyPlugin` when it starts.


## 5. Guide: Adding a New Component or System

This section walks through the process of adding new functionality to the DAM system.

### 5.1. Architectural Preference: Commands over Events/Stages

A key design principle for the `dam` ecosystem is to **prefer the Command pattern for implementing new functionality**.

-   **Why?**: Commands provide a clear, imperative, and traceable control flow. When you dispatch a command, you have a clear expectation of a specific action being performed. This makes the system easier to understand, debug, and test.
-   **Guideline**: Unless a task's requirements explicitly call for a decoupled, event-driven workflow (e.g., multiple independent systems reacting to a single occurrence) or a lifecycle-based stage, you should implement the logic as a command and its corresponding handler system. Avoid using events or component markers as the primary mechanism for triggering core business logic.

### 5.2. Guideline for New Systems

When adding a new system, first consider if it can be added to an existing plugin package (e.g., `dam_media_image`, `dam_psp`). If the new system provides functionality that is closely related to an existing plugin, it should be added to that plugin.

If the new system is not a good fit for an existing plugin, create a new plugin package for it. This keeps the codebase modular and allows for optional loading of functionality. (See "How to Create a New Plugin" for details).

### 5.3. Adding a New Component

The process for adding a new component is as follows:
1.  **Define the Component:** Create a new component class in the appropriate plugin package (e.g., `dam_media_image/models/`).
2.  **Register the Component:** Ensure the component is imported in the `__init__.py` of its package so that SQLAlchemy is aware of it.
3.  **Create a System:** Create a system to operate on the new component.
4.  **Register the System:** Register the system in your plugin's `build` method. The plugin itself is discovered via its entry point (see "How to Create a New Plugin").

### 5.4. Adding a New Command and Handler

The Command pattern is used for imperative actions where the caller requests a specific operation to be performed.

1.  **Define the Command:**
    - In the appropriate package, create a `commands.py` file if it doesn't exist.
    - Define a new dataclass that inherits from an appropriate base command in `dam.core.commands`. For tasks that analyze a single entity, `AnalysisCommand` is a good choice as it provides common fields and helpers.
    - Add any additional fields the command needs.

    *Example (`packages/my_plugin/commands.py`):*
    ```python
    from dataclasses import dataclass
    from dam.commands.analysis_commands import AnalysisCommand

    @dataclass
    class ExtractDominantColorCommand(AnalysisCommand[None]):
        """A command to extract the dominant color from an image asset."""
        # entity_id, depth, and stream are inherited from AnalysisCommand
        pass # No extra fields needed for this simple command
    ```

2.  **Create the Command Handler System:**
    - In the package's `systems/` module, create a new function to handle the command.
    - Decorate it with `@system(on_command=YourCommandClass)`.
    - Use the command's helper methods and the `WorldTransaction` object to implement the logic.

    *Example (`packages/my_plugin/systems/asset_systems.py`):*
    ```python
    from dam.core.systems import system
    from dam.core.transaction import WorldTransaction
    from dam.core.world import World
    from my_plugin.commands import ExtractDominantColorCommand
    from my_plugin.models import DominantColorComponent # A UniqueComponent

    @system(on_command=ExtractDominantColorCommand)
    async def handle_extract_dominant_color_command(
        cmd: ExtractDominantColorCommand,
        transaction: WorldTransaction, # Injected by the DI system
        world: World, # Needed for command helpers like open_stream
    ):
        print(f"Handling command to extract color from entity {cmd.entity_id}")

        # Use the command's helper to get a file stream.
        # This helper needs the 'world' to dispatch other commands if necessary.
        async with cmd.open_stream(world) as stream:
            if not stream:
                print(f"Could not get asset stream for entity {cmd.entity_id}")
                return

            # ... process the stream to find the dominant color (e.g., '#FF0000') ...
            dominant_color_hex = "#FF0000"  # Placeholder

        # Create the component instance
        color_component = DominantColorComponent(hex_color=dominant_color_hex)

        # The WorldTransaction object gives access to the session and other helpers.
        # Since DominantColorComponent is a UniqueComponent, this will correctly
        # add it if it's new, or update it if it already exists.
        await transaction.add_component_to_entity(cmd.entity_id, color_component)
        print(f"Dominant color for entity {cmd.entity_id} set to {dominant_color_hex}")
    ```

### 5.5. Interactive Command Handlers with Async Generators

For complex operations that require two-way communication with the caller (e.g., requesting user input, handling passwords), command handlers can be implemented as **async generators**.

-   **Yielding Events**: Instead of returning a single value, the handler can `yield` events (like `PasswordRequest`).
-   **Receiving Responses**: The `SystemExecutor` running the handler can send responses back into the generator using the `asend()` method.

1.  **Define a Command with Request/Response Types**:
    - Use `BaseCommand[ResultType, EventType]` to define the types.
    - `EventType` should be a `Union` of all events the handler might `yield`.
    - `ResultType` is the final return type (often `None` for interactive commands).

    *Example (`packages/my_plugin/commands.py`):*
    ```python
    from dam.commands.core import BaseCommand
    from dam.system_events.requests import PasswordRequest, PasswordResponse

    @dataclass
    class ProcessProtectedArchive(BaseCommand[None, Union[PasswordRequest, PasswordResponse]]):
        """A command that may require a password."""
        entity_id: int
    ```

2.  **Create the Interactive Handler**:
    - The handler's return type hint should be `AsyncGenerator[EventType, ResponseType]`.
    - `yield` events to the caller. The value sent back via `asend()` will be the result of the `yield` expression.

    *Example (`packages/my_plugin/systems/asset_systems.py`):*
    ```python
    from typing import AsyncGenerator, Union
    from dam.system_events.requests import PasswordRequest, PasswordResponse

    @system(on_command=ProcessProtectedArchive)
    async def handle_protected_archive(
        cmd: ProcessProtectedArchive,
    ) -> AsyncGenerator[Union[PasswordRequest, PasswordResponse], PasswordResponse]:
        password: str | None = None
        while not password:
            response = yield PasswordRequest(message="Password required")
            if response and response.password:
                # ... try to open archive with password ...
                is_correct = True # Placeholder
                if is_correct:
                    password = response.password
                    print("Archive unlocked!")
                else:
                    print("Incorrect password.")
            else:
                print("No password provided. Aborting.")
                break # Exit the loop
    ```

3.  **Dispatch and Interact with the Command**:
    - To interact with the command, you can use a `try...except StopAsyncIteration` block to handle the generator's lifecycle.

    *Example:*
    ```python
    command = ProcessProtectedArchive(entity_id=456)
    executor = world.dispatch_command(command)

    try:
        event = await anext(executor)
        while True:
            if isinstance(event, PasswordRequest):
                print(f"System requested password: {event.message}")
                password_input = "my-secret-password"
                event = await executor.asend(PasswordResponse(password=password_input))
    except StopAsyncIteration:
        pass
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
