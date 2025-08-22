# Developer Guide: ECS Digital Asset Management (DAM) System

## 1. Introduction

This document provides guidance for developers working on the ECS Digital Asset Management (DAM) system. This project implements a DAM using an Entity-Component-System (ECS) architecture in Python. The core idea is to represent assets (images, videos, audio, documents, etc.) as Entities (simple IDs), which are then described by attaching various Components (data-only dataclasses). Systems (typically service functions or dedicated modules) operate on entities based on the components they possess.

## 2. Core Architectural Concepts

The system is built upon the Entity-Component-System (ECS) pattern, which promotes flexibility and modularity.

### 2.1. Entities
-   **Definition**: Entities are unique identifiers (typically integers or UUIDs) representing a single digital asset or concept within the system. They don't hold data themselves but act as a central point to which Components are attached.
-   **Implementation**: In our system, Entities are represented by the `dam.models.entity.Entity` SQLAlchemy model, which primarily provides a unique `id`.

### 2.2. Components
-   **Definition**: Components are data-only objects that describe a specific aspect or property of an entity. Each component type defines a specific piece of data. Examples include:
    -   **Core File/Asset Descriptors**:
        -   `OriginalSourceInfoComponent`: Classifies the origin of the asset's content.
        - `FilePropertiesComponent`: Stores original filename and file size.
        -   `FileLocationComponent`: Stores the physical location of an entity's content.
        -   `ContentHashSHA256Component`, `ContentHashMD5Component`: Store content hashes.
    -   **Media-Specific Properties**:
        -   `ImageDimensionsComponent`: Stores width and height.
        -   `ImagePerceptualPHashComponent` (and AHash, DHash): Stores perceptual hashes for images.
        -   `AudioPropertiesComponent`: Stores metadata for audio tracks.
        -   `FramePropertiesComponent`: Stores metadata for frame sequences (videos, GIFs).
    -   **Conceptual Modeling & Versioning (New)**:
        -   `BaseConceptualInfoComponent` (Abstract): Base for defining conceptual works.
        -   `ComicBookConceptComponent`: Concrete example for comic book concepts (series, issue, year).
        -   `BaseVariantInfoComponent` (Abstract): Base for file variants of a conceptual work.
        -   `ComicBookVariantComponent`: Concrete example for comic book variants (language, format).
    -   **Tagging System (New)**:
        -   `TagConceptComponent`: Defines a tag, its scope, and properties (e.g., tag name "Sci-Fi", scope "GLOBAL"). This component is on an Entity that *is* the tag definition.
        -   `EntityTagLinkComponent`: Links any entity to a `TagConceptEntity`, effectively applying the tag, optionally with a value.
-   **Implementation**:
    -   Components inherit from `dam.models.base_component.BaseComponent`.
    - Dataclass behavior (including `kw_only=True`) is inherited from `dam.models.core.base_class.Base`.
    - Table names typically follow the convention `component_[name]`.
    - **Constructor Note**: For components inheriting from `BaseComponent`, the `entity_id` field and the `entity` relationship attribute are both `init=False`. This means you do not pass `entity` or `entity_id` when creating the component instance. Components should be instantiated with their own specific data fields. The linkage to an `Entity` (setting `entity_id` and the `entity` relationship) is managed by the `dam.services.ecs_service.add_component_to_entity` function after the component is created. For components not directly inheriting from `BaseComponent` (like association objects such as `PageLink`), their constructors follow standard SQLAlchemy model instantiation patterns based on their `MappedAsDataclass` definition.
    - **Model Registration**: Ensure all model modules (components, association objects like `PageLink`) are imported (e.g., in `dam/models/__init__.py`) so that their definitions are processed. This is crucial for SQLAlchemy's `Base.metadata` to be aware of all tables before database operations like `create_all()` or Alembic autogeneration.

### 2.3. BaseComponent
-   Provides common fields: `id`, `entity_id` (FK to `entities.id`), `created_at`, `updated_at`, and an `entity` relationship.

#### 2.3.1. Association Objects (New)
-   For ordered many-to-many relationships or relationships with extra data, direct SQLAlchemy models (inheriting from `Base`) are used instead of components.
-   Example: `PageLink` (in `dam/models/conceptual/page_link.py`) links an owner entity (e.g., a `ComicBookVariantComponent`'s entity) to page image entities, storing `page_number`.

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
    *   `WorldSession`: The active SQLAlchemy session for the current world (typically `Annotated[AsyncSession, "WorldSession"]`).
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

### 2.8. Services and Systems Imports

To support modularity and optional dependencies, the `dam.services` and `dam.systems` packages have a specific import design.

-   **Services**: Service modules (e.g., `dam.services.tag_service`) are self-contained and should be imported directly via their full path (e.g., `from dam.services import tag_service`). The `dam.services` package does not expose all services through its `__init__.py`. For services with heavy optional dependencies (like `semantic_service`), it is recommended to use local, on-demand imports within the functions that need them.

-   **Systems**: The `dam.systems` package automatically discovers and imports all modules within its directory at runtime. This means you can add a new system file to the `dam/systems/` directory and it will be automatically loaded without needing to edit any `__init__.py` file. The `dam.core.world_setup.py` file, which registers core systems, also uses try-except blocks to handle `ImportError`. This makes the system registration robust against missing optional dependencies.

### 2.9. Querying Entities with `ecs_service`

The `dam.services.ecs_service` module provides several helper functions to facilitate common queries for entities based on their components, reducing boilerplate and promoting optimized query patterns. These functions should be preferred for common query needs within systems or other services.

*   **`find_entities_with_components`**
    *   **Purpose**: Retrieves a list of distinct `Entity` objects that possess *all* of the specified component types.
    *   **Signature**: `async def find_entities_with_components(session: AsyncSession, required_component_types: List[Type[BaseComponent]]) -> List[Entity]`
    *   **Example**:
        ```python
        from dam.services import ecs_service
        from dam.models import FilePropertiesComponent, ImageDimensionsComponent
        from sqlalchemy.ext.asyncio import AsyncSession # Assuming AsyncSession is obtained

        # session: AsyncSession = ... obtain session ...
        image_entities = await ecs_service.find_entities_with_components( # Note: await
            session,
            [FilePropertiesComponent, ImageDimensionsComponent]
        )
        for entity in image_entities:
            # This entity has both FilePropertiesComponent and ImageDimensionsComponent
            pass
        ```

*   **`find_entities_by_component_attribute_value`**
    *   **Purpose**: Retrieves a list of distinct `Entity` objects that have a specific component where a particular attribute of that component matches a given value.
    *   **Signature**: `async def find_entities_by_component_attribute_value(session: AsyncSession, component_type: Type[T], attribute_name: str, value: Any) -> List[Entity]` (where `T` is a `BaseComponent` subclass)
    *   **Example**:
        ```python
        from dam.services import ecs_service
        from dam.models import FilePropertiesComponent
        from sqlalchemy.ext.asyncio import AsyncSession # Assuming AsyncSession is obtained

        # session: AsyncSession = ... obtain session ...
        jpeg_entities = await ecs_service.find_entities_by_component_attribute_value( # Note: await
            session,
            FilePropertiesComponent,
            "original_filename",
            "%.jpeg"
        )
        for entity in jpeg_entities:
            # This entity has a FilePropertiesComponent with original_filename ending in .jpeg
            pass
        ```
    *   **Performance Note**: For optimal performance with `find_entities_by_component_attribute_value`, ensure that attributes frequently used for querying are indexed in their respective component model definitions.

### 2.9. Asset Versioning, Structure, and Tagging (New Section)

The DAM system has been extended to support more complex relationships between assets, including versioning, structured content (like comic book pages), and a flexible tagging system.
#### 2.9.1. Conceptual Assets and Variants (Example: Comic Books)

This model allows grouping different file versions or manifestations under a common "conceptual work."

*   **Abstract Base Components:**
    *   `dam.models.conceptual.BaseConceptualInfoComponent`: An abstract base for components that define the *concept* of a work. Entities with such a component represent the abstract idea (e.g., "Amazing Spider-Man #1, 1963").
    *   `dam.models.conceptual.BaseVariantInfoComponent`: An abstract base for components that mark a file `Entity` as a specific *variant* or version of a conceptual work. It includes `conceptual_entity_id` to link to the concept's `Entity`.

*   **Concrete Comic Book Example:**
    *   **`ComicBookConceptComponent`**: Defined in `dam.models.conceptual.comic_book_concept_component.py`. Inherits `BaseConceptualInfoComponent`.
        *   Attached to an `Entity` that represents the abstract idea of a specific comic book (e.g., a particular issue or collected edition).
        *   Fields: `comic_title`, `series_title` (optional), `issue_number` (optional), `publication_year` (optional).
    *   **`ComicBookVariantComponent`**: Defined in `dam.models.conceptual.comic_book_variant_component.py`. Inherits `BaseVariantInfoComponent`.
        *   Attached to an `Entity` that represents an actual file (e.g., a PDF, CBZ, or high-resolution scan).
        *   Links to the `ComicBookConceptComponent`'s entity via the inherited `conceptual_entity_id`.
        *   Fields: `language` (optional), `format` (optional, e.g., "PDF", "CBZ"), `scan_quality` (optional), `is_primary_variant` (boolean), `variant_description` (optional).

*   **Managing Comic Book Concepts and Variants:**
    *   The `dam.services.comic_book_service.py` module provides specialized functions for these types:
        *   `create_comic_book_concept()`: Creates an entity with `ComicBookConceptComponent`.
        *   `link_comic_variant_to_concept()`: Adds `ComicBookVariantComponent` to a file entity and links it to a concept.
        *   Other functions for querying variants, finding concepts, setting primary variants, etc.

#### 2.9.2. Ordered Content (Example: Comic Book Pages)

To represent ordered sequences of images, like pages in a comic book.

*   **`PageLink` Association Object:** Defined in `dam.models.conceptual.page_link.py`. This is a SQLAlchemy model inheriting from `Base` (not `BaseComponent`).
    *   It creates a many-to-many relationship between an "owner" entity and "page image" entities, with an order.
    *   Fields:
        *   `owner_entity_id`: ForeignKey to `entities.id`. For comics, this is the ID of the `Entity` that has the `ComicBookVariantComponent`.
        *   `page_image_entity_id`: ForeignKey to `entities.id` (the `Entity` for the image file).
        *   `page_number`: Integer defining the order.
    *   This structure allows an image to be a page in multiple "owner" contexts (e.g., different comic variants) and supports bidirectional queries.

*   **Managing Comic Pages (via `dam.services.comic_book_service.py`):**
    *   `assign_page_to_comic_variant()`: Creates a `PageLink` record.
    *   `remove_page_from_comic_variant()`, `remove_page_at_number_from_comic_variant()`
    *   `get_ordered_pages_for_comic_variant()`: Retrieves page image `Entity` objects in order.
    *   `get_comic_variants_containing_image_as_page()`: Finds which comic variants use a specific image.
    *   `update_page_order_for_comic_variant()`: Replaces the entire page sequence for a variant.

#### 2.9.3. Tagging System

A flexible tagging system where tag definitions are themselves conceptual entities.

*   **`TagConceptComponent`**: Defined in `dam.models.conceptual.tag_concept_component.py`. Inherits from `BaseConceptualInfoComponent`.
    *   Attached to an `Entity` that represents the definition of a tag.
    *   Fields:
        *   `tag_name`: The name of the tag (e.g., "Sci-Fi", "Character:Spider-Man"). Usually unique.
        *   `tag_scope_type`: String defining the tag's applicability (e.g., "GLOBAL", "COMPONENT_CLASS_REQUIRED", "CONCEPTUAL_ASSET_LOCAL").
        *   `tag_scope_detail`: Extra information for the scope (e.g., a component class name, or an Entity ID of a conceptual asset for local scope).
        *   `tag_description`: Optional description of the tag.
        *   `allow_values`: Boolean indicating if this tag can be applied with a specific value (e.g., for a "Rating" tag, the value might be "5 Stars").

*   **`EntityTagLinkComponent`**: Defined in `dam.models.conceptual.entity_tag_link_component.py`. Inherits from `BaseComponent`.
    *   This component is attached to the `Entity` being tagged.
    *   Fields:
        *   `tag_concept_entity_id`: ForeignKey to the `Entity` that has the `TagConceptComponent` (the tag definition). `ondelete="CASCADE"` is set.
        *   `tag_value`: Optional string value for the tag application, used if `TagConceptComponent.allow_values` is true.
    *   Relationship `tag_concept` links back to the tag definition entity, with `passive_deletes=True`.

*   **Managing Tags (via `dam.services.tag_service.py`):**
    *   `create_tag_concept()`: Defines a new tag.
    *   `apply_tag_to_entity()`: Applies a defined tag to an entity. This function includes logic to validate the tag's scope against the target entity.
    *   `get_tags_for_entity()`: Retrieves all tags (and their values) applied to an entity.
    *   `get_entities_for_tag()`: Finds all entities that have a specific tag applied (optionally filtering by value).
    *   Other functions for updating and deleting tag definitions and applications.

This hybrid approach allows structured versioning and page management for specific types like comics, while the tagging system provides a flexible way to add arbitrary, scoped metadata across all types of entities.


### 2.10. Error Handling in ECS Operations

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

### 2.11. Performance Considerations
Internal optimizations have been implemented to enhance the performance of common ECS operations. Notably:
-   Fetching entities based on `MarkedEntityList` dependencies in systems is now more efficient, using optimized database queries.
-   The automatic removal of marker components by the `WorldScheduler` after system processing has been streamlined to reduce database overhead.
-   Indexing has been added to certain component attributes to speed up queries. Developers should continue to consider indexing for attributes frequently used in query conditions.

## 3. Project Structure

A brief overview of the key directories:

```
ecs_dam_system/
├── dam/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI application
# │   ├── gradio_ui.py        # Gradio UI code (Removed)
│   ├── models/             # SQLAlchemy models (Components & Association Objects)
│   │   ├── __init__.py
│   │   ├── core/
│   │   ├── conceptual/     # Models for conceptual assets, variants, pages, tags
│   │   │   ├── base_conceptual_info_component.py
│   │   │   ├── comic_book_concept_component.py
│   │   │   ├── base_variant_info_component.py
│   │   │   ├── comic_book_variant_component.py
│   │   │   ├── page_link.py
│   │   │   ├── tag_concept_component.py
│   │   │   └── entity_tag_link_component.py
│   │   └── ...
│   ├── services/           # Business logic
│   │   ├── __init__.py
│   │   ├── comic_book_service.py # Service for managing comic book concepts, variants, and pages
│   │   ├── tag_service.py      # Service for managing tags
│   │   └── ...
│   ├── systems/            # ECS Systems
│   └── core/               # Core ECS framework, DB session, settings
├── tests/                  # Pytest tests
│   ├── __init__.py
│   ├── test_comic_book_service.py
│   ├── test_tag_service.py # Tests for tagging
│   └── ...
└── ... (other project files)
```
(Ensure all new model and service files are accurately reflected here).

### 3.1. File Storage and Retrieval

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
    -   `FilePropertiesComponent` stores the `original_filename` and size.
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

For SQLAlchemy to recognize the new component model, for Alembic to generate migrations for it, and for `Base.metadata.create_all()` to create its table, the Python module defining the component class must be imported before these operations occur. This ensures the model class is evaluated and registers itself with `Base.metadata`.

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
    You must ensure that the module defining these components (e.g., `dam.core.components_markers`) is imported by a module that *is* loaded early, such as `dam/models/__init__.py` or your main application/test setup module before database operations. For instance, you could add `import dam.core.components_markers` to `dam/models/__init__.py`. A common practice is to have `dam/models/__init__.py` import all model modules to ensure they are registered with `Base.metadata`.

This explicit import step is vital for SQLAlchemy's declarative system to discover all models associated with `Base.metadata`.

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
The `dam.systems` package is configured to automatically discover and import all modules within its directory. Therefore, simply creating the new system file (e.g., `dam/systems/my_new_system.py`) is sufficient for it to be loaded at runtime. You do not need to manually add it to `dam/systems/__init__.py`.

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
            # Create the component instance with its specific data.
            # The BaseComponent's entity_id and entity fields are init=False,
            # so they are not passed to the constructor here.
            tag_component = TagComponent(tag_name=tag_name)

            # The ecs_service.add_component_to_entity function will handle:
            # 1. Setting tag_component.entity_id = entity.id
            # 2. Setting tag_component.entity = entity (the ORM relationship)
            # 3. Adding the component to the session.
            # The `entity` object must be the one managed by the current `db` session.
            # If `entity` was from a different session or detached, it might need to be db.get(Entity, entity.id).
            # Assuming `entity` is the correct, session-attached object:
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
-   **Current Status (Important):** Alembic is set up, but its usage for generating and applying migrations is **currently paused** during this phase of active schema evolution (related to conceptual assets, variants, pages, and tags).
-   **Development Database Setup:** For development, use the `dam-cli setup-db` command. This command will drop and recreate all tables based on the current SQLAlchemy model definitions. **This is destructive and only suitable for development environments.**
-   **Future Reactivation of Alembic:** Once the schema for these new features stabilizes, Alembic migrations will be re-introduced. The process will likely involve:
    1.  Clearing any old/obsolete migration files from `alembic/versions/`.
    2.  Ensuring `alembic/env.py` is correctly configured to target `Base.metadata` from `dam.models.core.base_class`.
    3.  Generating a new "baseline" migration that reflects the entire current schema: `alembic revision -m "baseline_schema_with_versioning_and_tagging" --autogenerate`.
    4.  Carefully reviewing the autogenerated script.
    5.  Applying this baseline to development databases: `alembic upgrade head`.
    From that point on, subsequent schema changes would again be managed by new incremental Alembic revisions.

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
    -   Entity Table: `entities`.
    -   Component Tables: Generally `component_[component_name]` (e.g., `component_file_location`, `component_tag`).
    -   Specific Hash Component Tables: `component_content_hash_[hashtype]` (e.g., `component_content_hash_sha256`) or `component_image_perceptual_hash_[hashtype]` (e.g., `component_image_perceptual_hash_phash`).
    -   Functions/Methods/Variables: `snake_case`.
    -   **Naming Conventions Update / Component & Model Constructors**:
        - Component Tables: `component_[component_name]` (e.g., `component_comic_book_concept`, `component_tag_concept`).
        - Association Object Tables: Plural, descriptive (e.g., `page_links`).
        - **Component & Model Constructors (Important due to `MappedAsDataclass`):**
            - **Components inheriting from `BaseComponent`**:
                - Recall that `BaseComponent` defines `entity_id: Mapped[int] = mapped_column(init=False)` and `entity: Mapped["Entity"] = relationship(init=False)`.
                - Therefore, when you instantiate a component that inherits from `BaseComponent`, you **do not** pass `entity` or `entity_id` to its constructor.
                - You only pass the data fields specific to that component (those that are `init=True`, which is the default for fields unless specified otherwise).
                - The linking to the parent `Entity` (setting `entity_id` and the `entity` relationship) is handled by the `dam.services.ecs_service.add_component_to_entity` function after the component instance is created.
                - Example:
                    ```python
                    # Correct instantiation for a component inheriting BaseComponent
                    my_comp = MyComponent(my_specific_field="some_value")
                    # ... then later, to associate with an entity:
                    # ecs_service.add_component_to_entity(session, target_entity.id, my_comp)
                    ```
                - Example for `ComicBookConceptComponent`:
                    ```python
                    # Assuming ComicBookConceptComponent has fields like comic_title, series_title
                    concept_comp = ComicBookConceptComponent(comic_title="The Amazing Example", series_title="Examples Vol. 1")
                    # ... then associate with a concept_entity:
                    # ecs_service.add_component_to_entity(session, concept_entity.id, concept_comp)
                    ```
            - **`EntityTagLinkComponent` Example**:
                - This component inherits from `BaseComponent`, so `entity` and `entity_id` (referring to the entity being tagged) are `init=False`.
                - It defines `tag_concept_entity_id: Mapped[int] = mapped_column(init=False)` and `tag_concept: Mapped["Entity"] = relationship(...)` (which is `init=True` by default as `init` is not specified on the relationship itself). It also has `tag_value: Mapped[str | None]`.
                - So, its constructor, as generated by `MappedAsDataclass`, would expect `tag_concept` (the `Entity` defining the tag) and optionally `tag_value`.
                - Correct Instantiation:
                    ```python
                    # entity_to_be_tagged = some entity instance
                    # tag_definition_entity = an entity instance that has TagConceptComponent
                    link_comp = EntityTagLinkComponent(tag_concept=tag_definition_entity, tag_value="example_value")
                    # ... then associate this link component with the entity being tagged:
                    # ecs_service.add_component_to_entity(session, entity_to_be_tagged.id, link_comp)
                    ```
            - **Association Objects (e.g., `PageLink`) or Models inheriting directly from `Base`**:
                - These do not inherit `entity_id` or `entity` from `BaseComponent`.
                - Their `__init__` signature is determined by `MappedAsDataclass` based on their own field definitions. Columns and relationship attributes not marked `init=False` become constructor arguments.
                - Example: `PageLink(owner_entity=owner_entity, page_image=page_image_entity, page_number=1)`. Here, `owner_entity` and `page_image` are likely relationship attributes that are `init=True` by default, and their corresponding FK ID columns (`owner_entity_id`, `page_image_entity_id`) would be `init=False` if they are populated via these relationships.
            - **Rule of Thumb:** When in doubt, inspect the model definition. If a Mapped attribute (column or relationship) has `init=False` explicitly set, do not pass it to `__init__`. If `init` is not specified, it defaults to `True` for `MappedAsDataclass` constructor generation. The goal is typically to pass relationship *objects* to the constructor when they are the primary means of establishing links (and are `init=True`), and let SQLAlchemy derive the foreign key ID values. For `BaseComponent` children, the `entity`/`entity_id` linkage is a special case handled post-instantiation by `ecs_service`.

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
