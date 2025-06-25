# Developer Guide: ECS Digital Asset Management (DAM) System

## 1. Introduction

This document provides guidance for developers working on the ECS Digital Asset Management (DAM) system. This project implements a DAM using an Entity-Component-System (ECS) architecture in Python. The core idea is to represent assets (images, videos, audio, documents, etc.) as Entities (simple IDs), which are then described by attaching various Components (data-only dataclasses). Systems (typically service functions or dedicated modules) operate on entities based on the components they possess.

## 2. Core Architectural Concepts

The system is built upon the Entity-Component-System (ECS) pattern, which promotes flexibility and modularity.

### 2.1. Entities
-   **Definition**: Entities are unique identifiers (typically integers or UUIDs) representing a single digital asset within the system. They don't hold data themselves but act as a central point to which Components are attached.
-   **Implementation**: In our system, Entities are represented by the `dam.models.entity.Entity` SQLAlchemy model, which primarily provides a unique `id`.

### 2.2. Components
-   **Definition**: Components are data-only objects that describe a specific aspect or property of an entity. Each component type defines a specific piece of data. For example, a `FileLocationComponent` stores where an asset's file is located. Content hash components like `ContentHashSHA256Component` store specific types of content hashes (e.g., SHA256). Similarly, perceptual hashes for images are stored in specific components like `ImagePerceptualPHashComponent` (for pHash), `ImagePerceptualAHashComponent` (for aHash), etc.
-   **Implementation**:
    -   Components are implemented as Python dataclasses that also serve as SQLAlchemy models. This is achieved by inheriting from `dam.models.base_component.BaseComponent`, which itself inherits from `dam.models.base_class.Base` (configured with `MappedAsDataclass`).
    -   Each component is defined in its own file within the `dam/models/` directory (e.g., `dam/models/content_hash_sha256_component.py`, `dam/models/image_perceptual_phash_component.py`).
    -   `MappedAsDataclass` from SQLAlchemy allows us to define models using dataclass syntax, making them concise and type-hint friendly.
    -   All components typically have an `entity_id` foreign key linking them back to an `Entity`.

### 2.3. BaseComponent
-   The `dam.models.base_component.BaseComponent` is an abstract base class that all concrete components should inherit from.
-   It provides common fields required by most components:
    -   `id`: A primary key for the component instance itself.
    -   `entity_id`: A foreign key to `entities.id`, linking the component to an asset.
    -   `created_at`, `updated_at`: Timestamps for tracking component record changes.
    -   `entity`: A SQLAlchemy relationship property to easily navigate from a component instance back to its parent `Entity` object.
-   Using `BaseComponent` ensures consistency and reduces boilerplate when defining new components.

### 2.4. Systems (Services)
-   **Definition**: Systems (often referred to as Services in this project's context) contain the logic that operates on entities based on the components they possess. For example, a service might find all entities with a specific hash, or generate thumbnails for entities that have an image component but no thumbnail component yet.
-   **Implementation**:
    *   Services are typically implemented as Python functions or classes within the `dam/services/` directory (e.g., `dam/services/asset_service.py`, `dam/services/file_operations.py`).
    *   They interact with the database via SQLAlchemy sessions to query for entities with certain components or to add/update components.

## 3. Project Structure

A brief overview of the key directories:

-   `dam/`: Main package for the DAM system.
    -   `core/`: Core functionalities like database session management (`database.py`) and application configuration (`config.py`).
    -   `models/`: Contains all SQLAlchemy model definitions, where each component is typically in its own file (e.g., `entity.py`, `base_component.py`, `content_hash_sha256_component.py`, `image_perceptual_phash_component.py`).
        -   `base_class.py`: Defines the ultimate `Base` for SQLAlchemy models, configured with `MappedAsDataclass`.
        -   `types.py`: Custom SQLAlchemy type annotations (e.g., for timestamps).
    -   `services/`: Houses the business logic (Systems/Services) that operate on entities and components.
    -   `cli.py`: Defines the Typer-based command-line interface for interacting with the DAM.
-   `alembic/`: Contains Alembic migration scripts and configuration for managing database schema changes.
-   `doc/`: Project documentation, including this guide.
-   `tests/`: Contains Pytest tests.
    -   `tests/models/`: Tests for individual component models.
    -   `tests/services/`: Tests for service logic.
    -   `tests/test_data/`: Sample data files for testing (e.g., sample images).
-   `pyproject.toml`: Project metadata, dependencies, and tool configurations (Ruff, MyPy, Pytest).
-   `.env.example`: Example environment variables file.
-   `README.md`: Main project README.

### 2.5. File Storage and Retrieval

The DAM employs a content-addressable storage strategy for asset files, managed by the `dam.services.file_storage` module.

-   **Content Hashing**: When a file is added, its content is read, and a SHA256 hash is computed. This hash serves as the primary identifier for the file's content.
-   **Storage Path**: Files are stored in a nested directory structure derived from their SHA256 hash. The base storage directory is defined by `ASSET_STORAGE_PATH` in the application settings (see `dam.core.config.settings`).
    -   For a file with hash `abcdef1234...`, it would be stored at a path like: `<ASSET_STORAGE_PATH>/ab/cd/abcdef1234...`.
    -   This structure helps to avoid having too many files in a single directory, which can be inefficient for some filesystems.
-   **`store_file` Function**: The `dam.services.file_storage.store_file(file_content: bytes, original_filename: str) -> str` function is responsible for:
    1.  Calculating the SHA256 hash of the `file_content`.
    2.  Determining the storage path based on this hash.
    3.  Creating the nested directories if they don't exist.
    4.  Writing the `file_content` to the path, using the full hash as the filename.
    5.  Returning the SHA256 hash (file identifier).
    -   If a file with the same content (and thus the same hash) is stored again, it will not create a duplicate; the existing file is effectively reused. The `original_filename` is not used for the storage path itself but can be stored in metadata components (like `FileLocationComponent`).
-   **`get_file_path` Function**: The `dam.services.file_storage.get_file_path(file_identifier: str) -> Path | None` function reconstructs the absolute path to a stored file given its `file_identifier` (SHA256 hash).
-   **`FileLocationComponent`**: This component (defined in `dam.models.file_location_component`, table name `component_file_location`) stores how to locate an asset's content.
    -   `file_identifier`: Stores the SHA256 hash returned by `store_file`.
    -   `storage_type`: Set to `"local_content_addressable"` for files managed by this strategy.
    -   `original_filename`: Can store the original name of the file as ingested, providing context.
-   **Benefits**:
    -   **Deduplication**: Files with identical content are stored only once, saving storage space.
    -   **Integrity**: The hash acts as a checksum; if the file on disk changes, its hash would no longer match the identifier.
    -   **Permanent Identifiers**: The file identifier (hash) is based on content, not a mutable filename or path.

This approach ensures that the actual asset files are managed robustly and efficiently. The `asset_service` uses these `file_storage` functions when adding new assets.

---

## 4. Guide: Adding a New Component

This section walks through the process of adding a new component to the DAM system. We'll use a hypothetical `TagComponent` as an example, which could be used to associate simple string tags with an asset entity.

### Step 1: Define the Component Dataclass Model

First, create a new Python file for your component in the `dam/models/` directory. For our example, this would be `dam/models/tag_component.py`.

The component class should:
- Inherit from `BaseComponent`.
- **Not** be decorated with `@dataclass` directly. Dataclass behavior (including `kw_only=True` for `__init__` methods) is automatically inherited because `BaseComponent`'s ultimate ancestor, `dam.models.base_class.Base`, is configured as a `MappedAsDataclass` with `kw_only=True`.
- Define a `__tablename__` for the database table.
- Define its specific data fields using `Mapped` type hints and SQLAlchemy's `mapped_column`.

**Example: `dam/models/tag_component.py`**

```python
from dataclasses import dataclass
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, UniqueConstraint

from .base_component import BaseComponent
# Note: Base is inherited via BaseComponent

# No @dataclass decorator needed here; it's inherited from Base via BaseComponent
class TagComponent(BaseComponent):
    __tablename__ = "component_tag" # Table names follow component_[name] convention

    # Inherited fields: id, entity_id, created_at, updated_at, entity relationship

    tag_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    # category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True) # Example optional field

    __table_args__ = (
        # Ensure an entity doesn't have the same tag multiple times.
        # If categories were used, the constraint might be (entity_id, category, tag_name).
        UniqueConstraint("entity_id", "tag_name", name="uq_tag_entity_name"),
    )

    def __repr__(self):
        return f"TagComponent(id={self.id}, entity_id={self.entity_id}, tag_name='{self.tag_name}')"
```

**Key points:**
- **Inheritance**: `class TagComponent(BaseComponent):`
- **Dataclass Behavior**: Inherited automatically from `Base` (which is a `MappedAsDataclass` configured with `kw_only=True`). No explicit `@dataclass` decorator is needed on `TagComponent` itself.
- **Table Name**: `__tablename__ = "component_tag"` (following the `component_[name]` convention).
- **Custom Fields**: `tag_name: Mapped[str] = mapped_column(...)` defines the actual data this component holds. We've added `index=True` as tags are likely to be queried.
- **Constraints**: `__table_args__` can define `UniqueConstraint`, `Index`, etc. Here, we prevent duplicate tags per entity.

### Step 2: Register the Component

For SQLAlchemy to recognize the new component model and for easier imports, add it to `dam/models/__init__.py`:

```python
# dam/models/__init__.py

# ... other imports ...
from .tag_component import TagComponent # Add this import

# ...

__all__ = [
    # ... other model names ...
    "TagComponent", # Add to __all__
]
```

### Step 3: Create Database Migration (Alembic)

Whenever you add or modify a model (which translates to a database table), you need to create a database migration using Alembic.

1.  **Generate a new revision script**:
    Open your terminal in the project root and run:
    ```bash
    alembic revision -m "add_tag_component_table" --autogenerate
    ```
    -   Alembic will inspect your models (via `env.py` which imports `Base.metadata`) and compare them to the current database schema (if one exists and is tracked by Alembic).
    -   It will generate a new script in `alembic/versions/` (e.g., `xxxxxxxxxxxx_add_tag_component_table.py`).

2.  **Inspect the generated script**:
    Open the newly generated migration file. It should contain Python code using `op.create_table()` for your new component and `op.drop_table()` in the `downgrade()` function. Verify that the columns, constraints, and indexes match your model definition.
    ```python
    # Example content in the generated migration script's upgrade() function:
    # op.create_table('component_tag',  # Reflects new table name convention
    #     sa.Column('entity_id', sa.Integer(), nullable=False),
    #     sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    #     # ... created_at, updated_at columns from BaseComponent ...
    #     sa.Column('tag_name', sa.String(length=255), nullable=False),
    #     sa.ForeignKeyConstraint(['entity_id'], ['entities.id'], ),
    #     sa.PrimaryKeyConstraint('id'),
    #     sa.UniqueConstraint('entity_id', 'tag_name', name='uq_tag_entity_name')
    # )
    # op.create_index(op.f('ix_component_tag_entity_id'), 'component_tag', ['entity_id'], unique=False) # Index name reflects table
    # op.create_index(op.f('ix_component_tag_tag_name'), 'component_tag', ['tag_name'], unique=False) # Index name reflects table
    ```
    **Note**: If autogeneration fails or produces an incorrect script (especially with complex model changes or custom types), you may need to manually edit the script or write it from scratch, similar to how the initial project migration was created. Refer to `alembic/versions/3e7c6290c313_manual_create_initial_schema.py` for an example of a manually written migration.

3.  **Apply the migration**:
    To update your database schema with the new table, run:
    ```bash
    alembic upgrade head
    ```
    This applies all pending migrations.

### Step 4: Implement Service Logic (Optional but Recommended)

While not strictly required for the component to exist, you'll typically want service functions to manage instances of your new component. These could reside in an existing service file (like `dam/services/asset_service.py`) or a new service file dedicated to this type of component or related functionality.

**Example: Adding a function to `asset_service.py`**

```python
# dam/services/asset_service.py
# ... other imports ...
from dam.models.tag_component import TagComponent

# ...

def add_tag_to_entity(session: Session, entity_id: int, tag_name: str) -> TagComponent:
    # Check if tag already exists for this entity to avoid duplicates (handled by DB constraint too)
    existing_tag_stmt = select(TagComponent).where(
        TagComponent.entity_id == entity_id,
        TagComponent.tag_name == tag_name
    )
    existing_tag = session.execute(existing_tag_stmt).scalar_one_or_none()
    if existing_tag:
        return existing_tag

    # Assuming entity object is not needed for kw_only if kw_only is from Base
    # and BaseComponent's __init__ takes entity_id.
    # If BaseComponent's __init__ also requires 'entity' object due to kw_only behavior:
    #   entity_obj = session.get(Entity, entity_id)
    #   if not entity_obj: raise ValueError("Entity not found")
    #   tag_component = TagComponent(entity_id=entity_id, entity=entity_obj, tag_name=tag_name)
    # else (current setup where tests pass entity and entity_id):
    #   This implies the __init__ requires both. For service logic, fetching the entity might be cleaner.

    entity_obj = session.get(Entity, entity_id)
    if not entity_obj:
        # Or handle error appropriately
        raise ValueError(f"Entity with id {entity_id} not found.")

    tag_component = TagComponent(
        entity_id=entity_id,
        entity=entity_obj, # Required by BaseComponent's __init__ due to kw_only behavior
        tag_name=tag_name
    )
    session.add(tag_component)
    # session.commit() # Typically, service functions add to session; CLI/caller commits.
    return tag_component

def get_tags_for_entity(session: Session, entity_id: int) -> list[TagComponent]:
    stmt = select(TagComponent).where(TagComponent.entity_id == entity_id)
    return session.execute(stmt).scalars().all()

```
*Note on `kw_only=True` and `__init__`*: The example above assumes that due to `kw_only=True` being active (either on `BaseComponent` or inherited from `Base`), the `BaseComponent`'s `__init__` method effectively requires both `entity_id` and the `entity` object itself, as discovered during testing. Service logic should fetch the `Entity` object if it needs to instantiate components directly.

### Step 5: Integrate with CLI or Application Logic

Once you have the component and optional service functions, you can integrate them into your application, for example, by updating or adding a CLI command.

**Example: Modifying `dam-cli add-asset` to accept tags**

```python
# dam/cli.py
# ... other imports ...
# from dam.services.asset_service import add_tag_to_entity # If defined

@app.command(name="add-asset")
def cli_add_asset(
    filepath_str: Annotated[str, typer.Argument(..., help="Path to the asset file.", ...)],
    tags: Annotated[Optional[str], typer.Option(help="Comma-separated tags for the asset (e.g., 'photo,animal,cat').")] = None,
):
    # ... (existing file processing and asset adding logic) ...
    db = SessionLocal()
    try:
        # ... (call asset_service.add_asset_file to get entity and created_new) ...
        # entity, created_new = asset_service.add_asset_file(...)

        if entity and tags:
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            for tag_name in tag_list:
                # Assuming add_tag_to_entity is available in asset_service
                asset_service.add_tag_to_entity(db, entity.id, tag_name)
            typer.echo(f"Added tags: {', '.join(tag_list)} to Entity ID {entity.id}")

        db.commit()
        # ... (success messages) ...
    # ... (error handling and db.close()) ...
```
This example shows how the `add-asset` command could be extended to parse a `--tags` option and use the `add_tag_to_entity` service function.

### Step 6: Write Tests

Finally, and crucially, write tests for your new component and any associated logic:
-   **Model Tests** (e.g., in `tests/models/test_tag_component.py`):
    -   Test creating an instance of `TagComponent`.
    -   Test adding, retrieving, updating (if applicable), and deleting `TagComponent` records from the database.
    -   Test any unique constraints (e.g., adding the same tag twice to the same entity should fail or be handled).
    -   Test the relationship to the `Entity`.
-   **Service Tests** (if you created service functions):
    -   Test the logic of `add_tag_to_entity`, `get_tags_for_entity`, etc.
-   **CLI Tests** (if you modified CLI commands):
    -   Use Typer's `CliRunner` to test that the CLI command behaves as expected (e.g., that providing `--tags` results in `TagComponent`s being created).

This comprehensive approach ensures your new component is well-defined, integrated, and reliable.

---

## 5. Other Development Aspects

### 5.1. Database Migrations (Alembic Workflow)

Beyond adding new component tables, Alembic is used for all schema changes:
-   **Modifying existing tables**: E.g., adding a column to a component.
    1.  Make the change in your SQLAlchemy model file (e.g., add a new `Mapped[]` attribute).
    2.  Generate a new revision: `alembic revision -m "add_new_column_to_my_component" --autogenerate`
    3.  Inspect the script: It should contain `op.add_column()`.
    4.  Apply: `alembic upgrade head`
-   **Creating new indexes or constraints**: Can also often be autogenerated or added manually to a revision script.
-   **Branching and Merging**: Alembic supports branching for more complex team workflows, though for simpler projects, a linear history is common.
-   **Downgrading**: Each revision script's `downgrade()` function should correctly reverse the `upgrade()` operations. Test downgrades if critical (`alembic downgrade <target_revision>`).
-   **Current Revision**: Check current DB revision: `alembic current`
-   **History**: View migration history: `alembic history`

Always ensure your `env.py` is correctly configured to see your `Base.metadata` for autogenerate to work effectively.

### 5.2. Running Tests

The project uses `pytest` for testing.
-   **Run all tests**:
    ```bash
    python -m pytest
    ```
    (Or simply `pytest` if your environment is set up for it).
-   **Run specific test files or tests**:
    ```bash
    python -m pytest tests/models/test_entity.py
    python -m pytest tests/services/test_asset_service.py::test_add_image_asset_creates_perceptual_hashes
    ```
-   **Test Coverage**: Consider using `pytest-cov` for measuring test coverage.
-   **Fixtures**: Database session fixtures (`db_session`) and test data fixtures are defined in `tests/conftest.py` and individual test files.

### 5.3. Code Style and Conventions

-   **Formatting**: The project is configured with Ruff for linting and formatting (see `pyproject.toml` under `[tool.ruff]`).
    -   Check formatting: `ruff check .`
    -   Apply formatting: `ruff format .`
-   **Type Checking**: MyPy is configured for static type checking (see `pyproject.toml` under `[tool.mypy]`).
    -   Run type checker: `mypy .`
-   **Imports**: Follow standard Python import ordering (e.g., standard library, then third-party, then local application imports), often managed by formatters like Ruff.
-   **Naming Conventions**:
    -   Models: `PascalCase` (e.g., `FileLocationComponent`).
    -   Entity Table: `entities`.
    -   Component Tables: Generally `component_[component_name]` (e.g., `component_file_location`, `component_tag`).
    -   Specific Hash Component Tables: `component_content_hash_[hashtype]` (e.g., `component_content_hash_sha256`) or `component_image_perceptual_hash_[hashtype]` (e.g., `component_image_perceptual_hash_phash`).
    -   Functions/Methods/Variables: `snake_case`.

Adhering to these practices helps maintain a clean, consistent, and understandable codebase.

### 5.4. Logging

The DAM system uses the standard Python `logging` module for operational messages, diagnostics, and error reporting.

**Configuration:**
- Logging is configured by the `dam.core.logging_config.setup_logging()` function.
- This setup is automatically called when the CLI application starts (`dam.cli.py`).
- By default, logs are output to `sys.stderr`.
- The default log format is: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`.
- The default logging level is `INFO`.

**Log Level Configuration:**
- The logging level can be controlled via the `DAM_LOG_LEVEL` environment variable.
- Supported values are standard Python logging level names (case-insensitive), e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- If `DAM_LOG_LEVEL` is not set or is invalid, it defaults to `INFO`.
  Example: `export DAM_LOG_LEVEL=DEBUG` (Linux/macOS) or `set DAM_LOG_LEVEL=DEBUG` (Windows).

**Usage in Modules:**
- To use logging within any module of the `dam` application, obtain a logger instance specific to that module:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  ```
- Then use the logger methods for output:
  - `logger.debug("Detailed information, typically of interest only when diagnosing problems.")`
  - `logger.info("Confirmation that things are working as expected.")`
  - `logger.warning("An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.")`
  - `logger.error("Due to a more serious problem, the software has not been able to perform some function.")`
  - `logger.exception("Similar to error, but automatically includes traceback information. Use within an except block.")`
  ```python
  try:
      # ... some operation ...
  except Exception as e:
      logger.exception(f"An error occurred during operation foo: {e}")
  ```

**Guideline:**
- **Always prefer using the logging framework over `print()` statements** for any non-trivial diagnostic messages, operational status, warnings, or errors within the library/application code (i.e., in `dam/services`, `dam/core`, etc.).
- `typer.echo()` and `typer.secho()` in `dam/cli.py` are acceptable for direct user feedback from CLI commands, as this is their intended purpose. Internal service logic, however, should use logging.
