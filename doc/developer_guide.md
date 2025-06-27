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
        -   `FilePropertiesComponent`: Stores original filename, file size, and MIME type.
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
    -   Dataclass behavior (including `kw_only=True`) is inherited from `dam.models.base_class.Base`.
    -   Table names follow `component_[name]`.
    -   **Constructor Note**: Pass the `Entity` object to the `entity` parameter. `entity_id` is `init=False` and populated via the relationship. For components not directly inheriting from `BaseComponent` (like association objects), their constructors are standard SQLAlchemy model constructors.
    -   **Model Registration**: Ensure all model modules (components, association objects like `PageLink`) are imported (e.g., in `dam/models/__init__.py`) to be registered with `AppBase.metadata` before DB operations like `create_all()` or Alembic autogeneration.

### 2.3. BaseComponent
-   Provides common fields: `id`, `entity_id` (FK to `entities.id`), `created_at`, `updated_at`, and an `entity` relationship.

### 2.4. Association Objects (New)
-   For ordered many-to-many relationships or relationships with extra data, direct SQLAlchemy models (inheriting from `Base`) are used instead of components.
-   Example: `PageLink` (in `dam/models/conceptual/page_link.py`) links an owner entity (e.g., a `ComicBookVariantComponent`'s entity) to page image entities, storing `page_number`.

### 2.5. Systems
-   (Content largely as before: async functions, `@system` decorator, stages, DI).

### 2.6. Resources and ResourceManager
-   (Content largely as before).

### 2.7. Marker Components
-   (Content largely as before).

### 2.8. System Stages
-   (Content largely as before).

### 2.9. Querying Entities with `ecs_service`
-   (Content largely as before, `find_entities_with_components`, `find_entities_by_component_attribute_value` are still relevant).

### 2.10. Asset Versioning, Structure, and Tagging (New Section)

The DAM system has been extended to support more complex relationships between assets, including versioning, structured content (like comic book pages), and a flexible tagging system.

#### 2.10.1. Conceptual Assets and Variants (Example: Comic Books)

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

#### 2.10.2. Ordered Content (Example: Comic Book Pages)

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

#### 2.10.3. Tagging System

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

### 2.11. Error Handling in ECS Operations
-   (Content largely as before: `StageExecutionError`, `EventHandlingError`).

### 2.12. Performance Considerations
-   (Content largely as before).

## 3. Project Structure

```
ecs_dam_system/
├── dam/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI application
│   ├── ui/                 # PyQt6 UI code (Optional)
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

## 4. Guide: Adding a New Component
- (This section remains largely the same but the example `TagComponent` is now superseded by the more complete tagging system. The general principles of creating a component still hold).

## 5. Other Development Aspects

### 5.1. Database Migrations (Alembic Workflow)
-   **Current Status (Important):** Alembic is set up, but its usage for generating and applying migrations is **currently paused** during this phase of active schema evolution (related to conceptual assets, variants, pages, and tags).
-   **Development Database Setup:** For development, use the `dam-cli setup-db` command. This command will drop and recreate all tables based on the current SQLAlchemy model definitions. **This is destructive and only suitable for development environments.**
-   **Future Reactivation of Alembic:** Once the schema for these new features stabilizes, Alembic migrations will be re-introduced. The process will likely involve:
    1.  Clearing any old/obsolete migration files from `alembic/versions/`.
    2.  Ensuring `alembic/env.py` is correctly configured to target `AppBase.metadata` from `dam.models.core.base_class`.
    3.  Generating a new "baseline" migration that reflects the entire current schema: `alembic revision -m "baseline_schema_with_versioning_and_tagging" --autogenerate`.
    4.  Carefully reviewing the autogenerated script.
    5.  Applying this baseline to development databases: `alembic upgrade head`.
    From that point on, subsequent schema changes would again be managed by new incremental Alembic revisions.

### 5.2. Running Tests
- (Content largely as before).

### 5.3. Code Style and Conventions
- (Content largely as before, but ensure any new conventions from the versioning/tagging system are noted if necessary, e.g., constructor patterns for new components).
    -   **Naming Conventions Update**:
        - Component Tables: `component_[component_name]` (e.g., `component_comic_book_concept`, `component_tag_concept`).
        - Association Object Tables: Plural, descriptive (e.g., `page_links`).
        - **Component & Model Constructors (Important due to `MappedAsDataclass`):**
            - For components inheriting from `BaseComponent` (which provides `entity_id: Mapped[int] = mapped_column(init=False)` and `entity: Mapped["Entity"] = relationship(...)` where the relationship is an init argument):
                - Always pass the parent `Entity` object to the `entity` parameter (e.g., `MyComponent(entity=actual_entity_object, other_field="value")`).
                - Other fields defined directly on the component (not inherited `init=False` fields) are passed as keyword arguments.
                - Example: `ComicBookConceptComponent(entity=concept_entity, comic_title="...")`.
                - Example: `EntityTagLinkComponent(entity=entity_to_tag, tag_concept=tag_concept_entity, tag_value="...")`. Here, `tag_concept` is the relationship attribute, and its corresponding FK `tag_concept_entity_id` is `init=False`.
            - For association objects (like `PageLink`) or models inheriting directly from `Base` (not `BaseComponent`):
                - The `__init__` signature is determined by `MappedAsDataclass`. Columns not marked `init=False` are constructor arguments.
                - Relationship attributes not marked `init=False` also become constructor arguments.
                - Foreign Key columns that are part of a primary key are often `init=False` by default if there's a corresponding relationship attribute that *is* an init argument.
                - Example: `PageLink(owner=owner_entity, page_image=page_image_entity, page_number=1)`. Here, `owner` and `page_image` are relationship attributes that are init arguments, and their corresponding FK ID columns (`owner_entity_id`, `page_image_entity_id`) are `init=False`.
            - **Rule of Thumb:** When in doubt, inspect the model definition. If a column has `init=False`, don't pass it to `__init__`. If a relationship attribute does not have `init=False`, it's likely an `__init__` argument. The goal is typically to pass relationship *objects* to the constructor when they are primary means of establishing links, and let SQLAlchemy derive the FK ID values.

### 5.4. Logging
- (Content largely as before).
