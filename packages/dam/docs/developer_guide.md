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

When defining a component, you must decide if an entity can have multiple instances of it or only one.

#### 4.3.1. Standard (Multi-Instance) Components

These are the default. An entity can have many of these components attached. For example, an entity could have multiple `EntityTagLinkComponent` components, linking it to various tags.

To create a standard component, simply inherit from `BaseComponent`.

```python
from dam.models.core import BaseComponent
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String

class MyStandardComponent(BaseComponent):
    __tablename__ = "my_standard_component"
    some_data: Mapped[str] = mapped_column(String)
```

#### 4.3.2. Unique Components

A unique component is a component of which there can be only **one** instance per entity. For example, an entity can only have one `MimeTypeComponent`. This uniqueness is enforced at both the application and database level.

To create a unique component, inherit from `UniqueComponentMixin` in addition to `BaseComponent`.

**Example: Simple Unique Component**

For a component that has no other table arguments, simply add the mixin.

```python
from dam.models.core import BaseComponent
from dam.models.core.component_mixins import UniqueComponentMixin
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer

class MyUniqueComponent(UniqueComponentMixin, BaseComponent):
    __tablename__ = "my_unique_component"
    some_value: Mapped[int] = mapped_column(Integer)
```

**Example: Unique Component with other Table Arguments**

If your component needs to define its own `__table_args__` (e.g., for other constraints), you must define them using the `@declared_attr` pattern so they can be combined with the constraint from the `UniqueComponentMixin`.

```python
from sqlalchemy import CheckConstraint, String
from sqlalchemy.orm import Mapped, mapped_column, declared_attr
from dam.models.core import BaseComponent
from dam.models.core.component_mixins import UniqueComponentMixin

class MyComplexUniqueComponent(UniqueComponentMixin, BaseComponent):
    __tablename__ = "my_complex_unique_component"

    name: Mapped[str] = mapped_column(String)
    value: Mapped[int] = mapped_column(Integer)

    @declared_attr.directive
    def __table_args__(cls):
        # Get the UniqueConstraint from the mixin
        mixin_args = UniqueComponentMixin.__table_args__(cls)

        # Define this component's specific constraints
        local_args = (
            CheckConstraint("value > 0", name="value_positive_check"),
        )

        # Return the combined tuple of all arguments
        return mixin_args + local_args
```

### 4.4. Adding a New Command and Handler

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
    - Decorate the function with `@system(on_command=YourCommandClass)`.
    - The function must be `async` and its first argument should be the command object.
    - Use functions to perform the business logic.

    *Example (`packages/my_plugin/systems/asset_systems.py`):*
    ```python
    from dam.core.systems import system
    from dam.core.transaction import EcsTransaction
    from my_plugin.commands import RenameAssetCommand
    from my_plugin.models import NameComponent # Assuming a component that stores the name

    @system(on_command=RenameAssetCommand)
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

3.  **Dispatch the Command:**
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
