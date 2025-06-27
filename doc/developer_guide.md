# Developer Guide: ECS Digital Asset Management (DAM) System

## 1. Introduction

This document provides guidance for developers working on the ECS Digital Asset Management (DAM) system. This project implements a DAM using an Entity-Component-System (ECS) architecture in Python. The core idea is to represent assets (images, videos, audio, documents, etc.) as Entities (simple IDs), which are then described by attaching various Components (data-only dataclasses). Systems (typically service functions or dedicated modules) operate on entities based on the components they possess.

## 2. Core Architectural Concepts

The system is built upon the Entity-Component-System (ECS) pattern, which promotes flexibility and modularity.

### 2.1. Entities
-   **Definition**: Entities are unique identifiers (typically integers or UUIDs) representing a single digital asset within the system. They don't hold data themselves but act as a central point to which Components are attached.
-   **Implementation**: In our system, Entities are represented by the `dam.models.entity.Entity` SQLAlchemy model, which primarily provides a unique `id`.

### 2.2. Components
-   **Definition**: Components are data-only objects that describe a specific aspect or property of an entity. Each component type defines a specific piece of data. Examples include:
    - `OriginalSourceInfoComponent`: Classifies the origin of the asset's content (e.g., local file, web download, reference) using its `source_type` field. See `dam.models.source_info.source_types` for defined types. It does not store filename or path directly.
    - `FilePropertiesComponent`: Stores the authoritative properties of an entity's file, such as its `original_filename`, `file_size_bytes`, and `mime_type` (`component_file_properties`). This is the primary source for the original filename.
    - `FileLocationComponent`: Stores the physical location of an entity's content (`component_file_location`). Key attributes:
        - `content_identifier`: Typically the SHA256 hash of the content.
        - `storage_type`: Indicates the nature of the location (e.g., `"dam_managed_storage"`, `"external_file_reference"`).
        - `physical_path_or_key`: The actual path (e.g., relative path in DAM storage, absolute external path for references).
        - `contextual_filename`: An optional filename associated with this specific location, useful if `physical_path_or_key` is a hash or generic ID. The primary original filename is in `FilePropertiesComponent`.
    - `ContentHashSHA256Component`: Stores SHA256 content hash as bytes (`component_content_hash_sha256`). Key attribute: `hash_value` (bytes).
    - `ImageDimensionsComponent`: Stores width and height for visual assets (`component_image_dimensions`).
    - `ImagePerceptualPHashComponent`: Stores pHash for images (`component_image_perceptual_hash_phash`).
    - `AudioPropertiesComponent`: Stores metadata for audio tracks (duration, codec, sample rate) (`component_audio_properties`).
    - `FramePropertiesComponent`: Stores metadata for sequences of frames like animated GIFs or video tracks (frame count, duration, frame rate) (`component_frame_properties`).
-   **Video Asset Conceptualization**: Videos are conceptualized as a combination of components:
    - An `Entity` representing the video.
    - `FileLocationComponent` and `FilePropertiesComponent` as standard.
    - One `ImageDimensionsComponent` for the video's resolution.
    - One `FramePropertiesComponent` for its visual track details (frame count, fps, visual duration).
    - One or more `AudioPropertiesComponent` instances for its audio tracks.
    - The dedicated `VideoPropertiesComponent` has been removed in favor of this composite model.
-   **Implementation**:
    -   Components inherit from `dam.models.base_component.BaseComponent`. The `kw_only=True` dataclass behavior is inherited from `dam.models.base_class.Base` (which is a `MappedAsDataclass`), so components do not need the `@dataclass(kw_only=True)` decorator themselves.
    -   Each component is defined in its own file within `dam/models/`.
    -   Table names strictly follow the `component_[name]` convention (e.g., `ImageDimensionsComponent` maps to `component_image_dimensions`).
    -   **Constructor Note**: When instantiating components that inherit from `BaseComponent`, provide the SQLAlchemy `Entity` object to the `entity` parameter. Do not pass `entity_id` directly to the constructor, as `entity_id` is marked `init=False` in `BaseComponent` and is populated via the `entity` relationship.
    -   **Model Registration for SQLAlchemy**: It's crucial that all SQLAlchemy models, including components defined outside the primary `dam/models/` directory (e.g., marker components in `dam/core/components_markers.py`), are imported at a point where they become registered with the shared `AppBase.metadata` object. This must happen before operations like `AppBase.metadata.create_all()` (often used in tests) or Alembic migration generation (`alembic revision --autogenerate`) are performed. This ensures their tables are correctly created and managed. This can be achieved by importing these model modules in `dam/models/__init__.py` or another central point that is loaded early in the application's lifecycle.

### 2.3. BaseComponent
-   The `dam.models.base_component.BaseComponent` is an abstract base class that all concrete components should inherit from.
-   It provides common fields required by most components:
    -   `id`: A primary key for the component instance itself.
    -   `entity_id`: A foreign key to `entities.id`, linking the component to an asset.
    -   `created_at`, `updated_at`: Timestamps for tracking component record changes.
    -   `entity`: A SQLAlchemy relationship property to easily navigate from a component instance back to its parent `Entity` object.
-   Using `BaseComponent` ensures consistency and reduces boilerplate when defining new components.

### 2.4. Systems
-   **Definition**: Systems encapsulate the logic that operates on entities possessing specific combinations of components. They are the primary way business logic and data transformations are implemented in the ECS architecture.
-   **Implementation**:
    *   Systems are Python functions (typically `async def`) decorated with `@dam.core.systems.system(stage=SystemStage.SOME_STAGE)`.
    *   They are organized into modules within the `dam/systems/` directory (e.g., `dam/systems/metadata_systems.py`).
    *   Systems declare their dependencies (e.g., database session, configuration, resources, specific entity lists) using `typing.Annotated` type hints in their parameters.
-   **Execution**:
    *   The `dam.core.systems.WorldScheduler` is responsible for executing registered systems.
    *   Execution is typically organized into `SystemStage`s (see below). The scheduler runs all systems registered for a particular stage.
    *   (Future: Systems may also be triggered by events).
-   **Dependency Injection**: The `WorldScheduler` automatically injects dependencies into systems based on their annotated parameters. Common injectable types include:
    *   `WorldSession`: The active SQLAlchemy session for the current world (typically `Annotated[Session, "WorldSession"]`).
    *   `WorldName`: The string name of the current world (typically `Annotated[str, "WorldName"]`).
    *   `CurrentWorldConfig`: The `WorldConfig` object for the current world (typically `Annotated[WorldConfig, "CurrentWorldConfig"]`).
    *   `WorldContext`: The entire `WorldContext` object, providing access to session, world name, and world config. Can be requested directly via `my_context: WorldContext`.
    *   `Resource[ResourceType]`: An instance of a registered shared resource (see Section 2.5).
    *   `MarkedEntityList[MarkerComponentType]`: A list of `Entity` objects that have the specified `MarkerComponentType` attached.

### 2.5. Resources and ResourceManager
-   **Definition**: Resources are shared objects or services that systems might need to perform their tasks. Examples include file operation utilities, external API clients, etc.
-   **`ResourceManager`**: The `dam.core.resources.ResourceManager` is a central container for managing instances of these resources.
    *   Resources are typically instantiated once (e.g., at application startup) and added to the `ResourceManager`.
    *   Systems can request a resource by type-hinting a parameter: `my_file_ops: Annotated[FileOperationsResource, "Resource"]`.
-   **`FileOperationsResource`**: An example is `dam.core.resources.FileOperationsResource`, which wraps functions from `dam.services.file_operations` to make them injectable.

### 2.6. Marker Components
-   **Definition**: Marker components are special, often data-less, components used to tag or mark entities. They signal that an entity is in a particular state or requires specific processing.
-   **Usage**:
    *   A service (like `asset_service` during initial ingestion) might add a marker like `NeedsMetadataExtractionComponent` to an entity.
    *   A system (like `MetadataExtractionSystem`) can then request a `MarkedEntityList[NeedsMetadataExtractionComponent]` to get all entities that need metadata extraction.
    *   After processing, the system (or the scheduler) typically removes the marker or replaces it with another (e.g., `MetadataExtractedComponent`).
-   **Location**: Defined in `dam/core/components_markers.py`.

### 2.7. System Stages
-   **Definition**: `dam.core.stages.SystemStage` is an `Enum` that defines distinct phases or stages in the application's processing lifecycle (e.g., `ASSET_INGESTION`, `METADATA_EXTRACTION`, `POST_PROCESSING`).
-   **Usage**:
    *   Systems are registered to run at a specific stage using the `@system(stage=SystemStage.SOME_STAGE)` decorator.
    *   The `WorldScheduler.execute_stage(stage, world_context)` method executes all systems registered for that particular stage.
    *   This provides a way to order operations and manage dependencies between different processing steps.

### 2.8. Querying Entities with `ecs_service`

The `dam.services.ecs_service` module provides several helper functions to facilitate common queries for entities based on their components, reducing boilerplate and promoting optimized query patterns. These functions should be preferred for common query needs within systems or other services.

*   **`find_entities_with_components`**
    *   **Purpose**: Retrieves a list of distinct `Entity` objects that possess *all* of the specified component types.
    *   **Signature**: `find_entities_with_components(session: Session, required_component_types: List[Type[BaseComponent]]) -> List[Entity]`
    *   **Example**:
        ```python
        from dam.services import ecs_service
        from dam.models import FilePropertiesComponent, ImageDimensionsComponent
        from sqlalchemy.orm import Session # Assuming session is obtained

        # session: Session = ... obtain session ...
        image_entities = ecs_service.find_entities_with_components(
            session,
            [FilePropertiesComponent, ImageDimensionsComponent]
        )
        for entity in image_entities:
            # This entity has both FilePropertiesComponent and ImageDimensionsComponent
            pass
        ```

*   **`find_entities_by_component_attribute_value`**
    *   **Purpose**: Retrieves a list of distinct `Entity` objects that have a specific component where a particular attribute of that component matches a given value.
    *   **Signature**: `find_entities_by_component_attribute_value(session: Session, component_type: Type[T], attribute_name: str, value: Any) -> List[Entity]` (where `T` is a `BaseComponent` subclass)
    *   **Example**:
        ```python
        from dam.services import ecs_service
        from dam.models import FilePropertiesComponent
        from sqlalchemy.orm import Session # Assuming session is obtained

        # session: Session = ... obtain session ...
        jpeg_entities = ecs_service.find_entities_by_component_attribute_value(
            session,
            FilePropertiesComponent,
            "mime_type",
            "image/jpeg"
        )
        for entity in jpeg_entities:
            # This entity has a FilePropertiesComponent with mime_type 'image/jpeg'
            pass
        ```
    *   **Performance Note**: For optimal performance with `find_entities_by_component_attribute_value`, ensure that attributes frequently used for querying (like `mime_type` in the example above) are indexed in their respective component model definitions (e.g., `mime_type: Mapped[Optional[str]] = mapped_column(String(128), index=True)`). An index has been added to `FilePropertiesComponent.mime_type` as part of recent optimizations.

### 2.9. Error Handling in ECS Operations

When systems are executed via `World.execute_stage(...)` or event handlers via `World.dispatch_event(...)`, failures within the systems/handlers or during the final database commit will now result in specific custom exceptions being raised. This allows calling code to more effectively respond to operational failures. These exceptions are defined in `dam.core.exceptions`.

*   **`StageExecutionError(DamECSException)`**
    *   Raised by `World.execute_stage(...)` if a system within the stage fails or if the session commit for the stage fails.
    *   Key Attributes:
        *   `message: str`: General error message.
        *   `stage_name: str`: Name of the stage that failed.
        *   `system_name: Optional[str]`: Name of the specific system that caused the failure, if applicable.
        *   `original_exception: Optional[Exception]`: The underlying Python exception that was caught.

*   **`EventHandlingError(DamECSException)`**
    *   Raised by `World.dispatch_event(...)` if an event handler fails or if the session commit for the event dispatch fails.
    *   Key Attributes:
        *   `message: str`: General error message.
        *   `event_type: str`: Name of the event type being handled.
        *   `handler_name: Optional[str]`: Name of the specific handler function that failed, if applicable.
        *   `original_exception: Optional[Exception]`: The underlying Python exception that was caught.

*   **Example Usage**:
    ```python
    from dam.core.world import World # Assuming World object is available
    from dam.core.stages import SystemStage
    from dam.core.exceptions import StageExecutionError, EventHandlingError, DamECSException
    import logging # For example logging

    logger = logging.getLogger(__name__)

    # my_world: World = get_world(...) # Obtain your world instance

    async def run_my_stage(my_world: World):
        try:
            await my_world.execute_stage(SystemStage.METADATA_EXTRACTION)
            logger.info(f"Stage {SystemStage.METADATA_EXTRACTION.name} completed successfully.")
        except StageExecutionError as e:
            logger.error(f"Stage execution failed: {e.message}")
            logger.error(f"  Stage: {e.stage_name}")
            if e.system_name:
                logger.error(f"  Failing system: {e.system_name}")
            if e.original_exception:
                logger.error(f"  Original error: {type(e.original_exception).__name__}: {e.original_exception}")
            # Add specific handling, e.g., retry logic, marking entities as failed, etc.
        except DamECSException as e: # Catch other potential general ECS errors
            logger.error(f"An ECS operation failed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    ```
*   This improved error propagation ensures that failures in scheduled ECS operations are not silent and can be handled appropriately by the parts of the application orchestrating these processes (e.g., CLI commands, service layers).

### 2.10. Performance Considerations
Internal optimizations have been implemented to enhance the performance of common ECS operations. Notably:
-   Fetching entities based on `MarkedEntityList` dependencies in systems is now more efficient, using optimized database queries.
-   The automatic removal of marker components by the `WorldScheduler` after system processing has been streamlined to reduce database overhead.
-   Indexing has been added to certain component attributes (e.g., `FilePropertiesComponent.mime_type`) to speed up queries. Developers should continue to consider indexing for attributes frequently used in query conditions.

## 3. Project Structure

A brief overview of the key directories:

-   `dam/`: Main package for the DAM system.
    -   `core/`: Core ECS framework functionalities.
        -   `config.py`: Application settings (Pydantic).
        -   `database.py`: SQLAlchemy engine, session setup, `DatabaseManager`.
        -   `logging_config.py`: Logging setup.
        -   `systems.py`: `@system` decorator, `WorldScheduler`.
        -   `stages.py`: `SystemStage` enum.
        -   `resources.py`: `ResourceManager` and base `Resource` definitions.
        -   `system_params.py`: `Annotated` types for system dependency injection (e.g., `WorldSession`, `Resource`, `MarkedEntityList`).
        -   `components_markers.py`: Definitions for marker components (e.g., `NeedsMetadataExtractionComponent`).
    -   `models/`: Contains all SQLAlchemy model definitions for Entities and Components.
        -   `base_class.py`, `base_component.py`, `entity.py`, individual component files.
    -   `services/`: Contains helper services, often wrapped by or used within Systems, or for direct CLI actions not yet converted to systems. (e.g., `asset_service.py`, `file_storage.py`, `ecs_service.py`).
    -   `systems/`: Contains ECS System implementations.
        -   `metadata_systems.py`: Example system for metadata extraction.
    -   `cli.py`: Defines the Typer-based command-line interface.
-   `alembic/`: Contains Alembic migration scripts.
-   `doc/`: Project documentation.
-   `tests/`: Pytest tests.
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
    -   If a file with the same content (and thus the same hash) is stored again, it will not create a duplicate; the existing file is effectively reused. The `original_filename` (from `FilePropertiesComponent`) is not used for the storage path itself.
-   **`get_file_path` Function**: The `dam.services.file_storage.get_file_path(file_identifier: str) -> Path | None` function reconstructs the absolute path to a stored file given its `file_identifier` (SHA256 hash).
-   **`FileLocationComponent`**: This component (defined in `dam.models.core.file_location_component`, table name `component_file_location`) stores how to locate an entity's content.
    -   `content_identifier`: Stores the SHA256 hash (hex string) of the content.
    -   `storage_type`: Indicates how the asset is stored (e.g., `"dam_managed_storage"` for Content Addressable Storage in the local DAM, or `"external_file_reference"` for files stored by their original path).
    -   `physical_path_or_key`: The relative path within the DAM's CAS store (e.g., `ab/cd/hashvalue`) or the absolute original file path for references.
    -   `contextual_filename`: Can store an optional filename relevant to this specific location (e.g., if `physical_path_or_key` is a hash). The primary original filename for the asset is stored in `FilePropertiesComponent.original_filename`.
-   **Relationship with `OriginalSourceInfoComponent` and `FilePropertiesComponent`**:
    -   `OriginalSourceInfoComponent` classifies the source (e.g., local, web, reference) via `source_type`.
    -   `FilePropertiesComponent` stores the `original_filename`, size, and MIME type.
    -   `FileLocationComponent` stores the path, which could be a DAM-internal path or an external reference path.
-   **Benefits**:
    -   **Deduplication**: Files with identical content are stored only once in DAM-managed CAS, saving storage space.
    -   **Integrity**: The hash acts as a checksum.
    -   **Permanent Identifiers**: The content identifier (hash) is based on content, not a mutable filename or path.

This approach ensures that the actual asset files are managed robustly and efficiently. Systems like `asset_lifecycle_systems` use these `file_storage` functions and create `FileLocationComponent` entries.

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

### Step 2: Ensure Component Model Registration with SQLAlchemy Metadata

For SQLAlchemy to recognize the new component model, for Alembic to generate migrations for it, and for `AppBase.metadata.create_all()` to create its table, the Python module defining the component class must be imported before these operations occur. This ensures the model class is evaluated and registers itself with `AppBase.metadata`.

-   **If the component is defined within the `dam/models/` directory** (e.g., `dam/models/tag_component.py`):
    Add an import for it in `dam/models/__init__.py` and include its class name in the `__all__` list. For example:
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
-   **If the component is defined outside `dam/models/`** (e.g., marker components in `dam/core/components_markers.py`):
    You must ensure that the module defining these components (e.g., `dam.core.components_markers`) is imported by a module that *is* loaded early, such as `dam/models/__init__.py` or your main application/test setup module before database operations. For instance, you could add `import dam.core.components_markers` to `dam/models/__init__.py`. A common practice is to have `dam/models/__init__.py` import all model modules to ensure they are registered with the `AppBase.metadata`.

This explicit import step is vital for SQLAlchemy's declarative system to discover all models associated with `AppBase.metadata`.

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

### Step 4: Implement System Logic (Recommended)

While not strictly required for the component to exist, you'll typically want a System to manage or react to instances of your new component.

**Example: Creating a System to process entities with `TagComponent` (conceptual)**

Imagine you want a system that processes entities that have been tagged (e.g., logs them or performs another action). You might first add a marker, `NeedsTagProcessingComponent`, when a tag is added, and then have a system react to that.

Alternatively, if the system is to *add* tags based on some criteria, it might look like this (this example is more about *using* tags, but illustrates system structure):

Create `dam/systems/tag_processing_system.py`:
```python
from typing import List, Annotated
from dam.core.components_markers import NeedsTagProcessingComponent # Hypothetical marker
from dam.core.stages import SystemStage
from dam.core.system_params import WorldSession
from dam.core.systems import system
from dam.models import Entity
from dam.models.tag_component import TagComponent # Your new component
from dam.services import ecs_service # For direct component access

@system(stage=SystemStage.POST_PROCESSING) # Or an appropriate stage
async def process_tagged_entities_system(
    session: WorldSession,
    entities_with_marker: Annotated[List[Entity], "MarkedEntityList", NeedsTagProcessingComponent]
):
    if not entities_with_marker:
        return

    print(f"TagProcessingSystem: Found {len(entities_with_marker)} entities to process tags for.")
    for entity in entities_with_marker:
        tags = ecs_service.get_components(session, entity.id, TagComponent)
        tag_names = [tag.tag_name for tag in tags]
        print(f"Entity {entity.id} has tags: {tag_names}")
        # ... perform some action based on tags ...

        # Remove marker after processing
        marker = ecs_service.get_component(session, entity.id, NeedsTagProcessingComponent)
        if marker:
            ecs_service.remove_component(session, marker)
    # Session commit/flush is handled by the WorldScheduler per stage
```
Remember to import `tag_processing_system` in `dam/cli.py` or `dam/systems/__init__.py` to register it.

### Step 5: Integrate with CLI or Application Logic

How you integrate depends on the system's purpose:
-   **If adding components via CLI**: The CLI command might directly use `ecs_service.add_component_to_entity` or a helper service function. If complex logic or further processing is needed after adding the component, the CLI might add a *marker component* to the entity, and a dedicated system (like the example above) would pick it up in a later stage scheduled by the `WorldScheduler`.

**Example: Modifying `dam-cli add-asset` to accept tags and add `TagComponent` directly (simpler case)**
(Assuming `asset_service` has a helper `add_tag_to_entity_sync` for direct synchronous use by CLI if needed, or CLI manages session and calls `ecs_service` directly)

```python
# dam/cli.py
# ... other imports ...
from dam.models.tag_component import TagComponent # Import your new component
# from dam.services.asset_service import add_tag_to_entity_sync # Hypothetical sync helper

@app.command(name="add-asset")
def cli_add_asset(
    # ... other parameters ...
    tags: Annotated[Optional[str], typer.Option(help="Comma-separated tags (e.g., 'photo,animal,cat').")] = None,
):
    # ... (existing file processing and asset adding logic to get 'entity') ...
    # This part runs within a 'with db_manager.get_db_session(...) as db:' block in the actual CLI
    # So 'db' (the session) is available.

    # entity, created_new = asset_service.add_asset_file(...) # This gets the entity

    if entity and tags:
        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        for tag_name in tag_list:
            # Direct component addition (if no complex post-processing system is needed for this action)
            # Ensure entity object is loaded if `TagComponent` requires it for relationships
            entity_obj_for_tag = db.get(Entity, entity.id) # Re-fetch or ensure it's in session correctly
            if entity_obj_for_tag:
                tag_component = TagComponent(
                    entity_id=entity.id,
                    entity=entity_obj_for_tag, # BaseComponent requires this
                    tag_name=tag_name
                )
                # Use ecs_service to handle potential duplicates if not relying on DB constraint alone before flush
                ecs_service.add_component_to_entity(db, entity.id, tag_component, flush=False)
        typer.echo(f"Added tags: {', '.join(tag_list)} to Entity ID {entity.id}")
        # db.commit() is handled by the main context manager for the command
    # ...
```
If adding a tag should trigger further complex processing, the CLI would instead add a `NeedsTaggingProcessingComponent(tag_to_add="...")` and a system would handle the actual `TagComponent` creation and other logic.

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

The project uses `pytest` for testing, preferably run via `uv`.
-   **Run all tests**:
    ```bash
    uv run pytest
    ```
-   **Run specific test files or tests**:
    ```bash
    uv run pytest tests/models/test_entity.py
    uv run pytest tests/services/test_asset_service.py::test_add_image_asset_creates_perceptual_hashes
    ```
-   **Test Coverage**: Use `pytest-cov` (included in `[dev]` dependencies).
    ```bash
    uv run pytest --cov=dam --cov-report=term-missing
    ```
-   **Fixtures**: Database session fixtures (`db_session`, `test_db_manager`), application settings overrides (`settings_override`), and test data fixtures are defined in `tests/conftest.py` and individual test files.

### 5.3. Code Style and Conventions

-   **Formatting & Linting**: The project uses Ruff (see `pyproject.toml` under `[tool.ruff]`).
    -   Format code: `uv run ruff format .`
    -   Lint and apply auto-fixes: `uv run ruff check . --fix`
    -   Check for lint errors (without fixing): `uv run ruff check .`
-   **Type Checking**: MyPy is configured (see `pyproject.toml` under `[tool.mypy]`).
    -   Run type checker: `uv run mypy .`
-   **System Registration & Execution**:
    System functions are defined with decorators (`@system` or `@listens_for`) which collect their metadata (parameters, target stage/event) globally when modules are imported. However, for a system to be active within a specific `World`, it must be explicitly registered to that `World`'s scheduler via `world.register_system(...)`.
    The application's core systems are registered through the `dam.core.world_setup.register_core_systems(world_instance)` helper function (this function was moved from `world_registrar` to `world_setup`). This function is called when `World` instances are initialized, for example, by the CLI or in test setups. If you are developing a new system intended to be part of the standard set for all worlds, you should add its registration call to `register_core_systems`.
    For systems that are highly specific to a particular workflow or a custom `World` setup not managed by the default initialization, you would call `world.register_system(...)` manually after obtaining or creating your `World` instance.
    The `WorldScheduler` then executes these registered systems at defined stages or in response to events.

    In addition to stage-based and event-driven execution, a `World` instance can also execute a single system function on-demand using the `world.execute_one_time_system(system_func, session=optional_session, **kwargs)` method. This is useful for invoking specific system logic outside the standard stage/event flow, providing any necessary parameters via `kwargs`. The method handles dependency injection and session management.

-   **World Initialization and Resource Management**:
    -   A `World` instance is minimally initialized with its `WorldConfig` (in `World.__init__`). This includes creating an empty `ResourceManager` and a `WorldScheduler` that holds a reference to this (initially empty) manager.
    -   The essential resources (like `DatabaseManager`, `FileStorageService`, `WorldConfig` itself, `FileOperationsResource`) are then populated into the `World`'s `ResourceManager` by an external setup function, typically `dam.core.world.create_and_register_world`. This function calls `dam.core.world_setup.initialize_world_resources(world_instance)`, which modifies the `world_instance.resource_manager` in place.
    -   The `create_and_register_world` function also ensures the `world.scheduler.resource_manager` reference points to the now-populated resource manager (though this is often the same instance, it's a safeguard).
    -   This approach keeps `World.__init__` clean and centralizes the resource population and core system registration logic in `dam.core.world_setup` and the world creation functions.

-   **Imports**: Follow standard Python import ordering (e.g., standard library, then third-party, then local application imports), often managed by formatters like Ruff.
-   **Naming Conventions**:
    -   Models: `PascalCase` (e.g., `FileLocationComponent`).
    -   Component Instantiation: When creating component instances that inherit from `BaseComponent` (which uses `kw_only=True` dataclass behavior from `dam.models.base_class.Base`):
        - Foreign key ID fields (e.g., `entity_id` in `BaseComponent`, `website_entity_id` in `WebSourceComponent`) are typically required keyword arguments in the `__init__` constructor, unless they have a default or are marked `init=False` (which is not the case for these FK IDs).
        - SQLAlchemy relationship properties (e.g., `entity` in `BaseComponent`, `website` in `WebSourceComponent`) should be marked with `init=False` in their `relationship()` definition. This means they are NOT constructor arguments and are populated by the ORM after the instance is created and associated with a session (usually upon flush, based on the FK ID values).
        - Example: `my_comp = MyComponent(entity_id=some_entity.id, other_field='value')`.
        - The `ecs_service.add_component_to_entity` helper correctly handles linking the component to an entity object after instantiation.
    -   Entity Table: `entities`.
    -   Component Tables: Generally `component_[component_name]` (e.g., `component_file_location`, `component_tag`).
    -   Specific Hash Component Tables: `component_content_hash_[hashtype]` (e.g., `component_content_hash_sha256`) or `component_image_perceptual_hash_[hashtype]` (e.g., `component_image_perceptual_hash_phash`).
    -   Functions/Methods/Variables: `snake_case`.
    -   Component Constructors: When creating components derived from `BaseComponent`, remember that `entity_id` is `init=False`. You should pass the `Entity` object itself to the `entity` parameter (e.g., `MyComponent(entity=actual_entity_object, other_field="value")`). The `entity_id` will be populated by SQLAlchemy through this relationship.

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
