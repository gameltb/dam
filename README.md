# ECS Digital Asset Management (DAM) System

This project implements a Digital Asset Management system using an Entity-Component-System (ECS) architecture in Python. The core idea is to represent assets (images, videos, audio, documents, etc.) as Entities (simple IDs), which are then described by attaching various Components (data-only dataclasses). Systems (typically service functions or dedicated modules) operate on entities based on the components they possess.

## Key Technologies & Concepts

*   **SQLAlchemy ORM**: Used for database interaction, with `MappedAsDataclass` to define components as Python dataclasses that are also database models.
*   **Alembic**: Manages database schema migrations (currently paused during active schema refactoring).
*   **ECS Core Framework**:
    *   **Entities, Components, Systems**: Standard ECS pattern.
    *   **WorldScheduler**: Manages system execution.
    *   **Dependency Injection**: For systems.
    *   **Resources**: Shared utilities.
*   **Modularity**: Components in `dam/models/`, services in `dam/services/`, systems in `dam/systems/`.
*   **Content-Addressable Storage (CAS)**: Files stored by hash.
*   **Metadata Extraction**: For various file types.
*   **Asset Versioning & Structure**: Flexible model for grouping versions of a conceptual work and defining ordered content (e.g., comic pages). Uses:
    *   Abstract bases: `BaseConceptualInfoComponent`, `BaseVariantInfoComponent`.
    *   Concrete components: e.g., `ComicBookConceptComponent`, `ComicBookVariantComponent`.
    *   Association objects: e.g., `PageLink` for ordered pages.
*   **Tagging System**: Tags are defined as conceptual assets (`TagConceptComponent`) and applied to entities via a link component (`EntityTagLinkComponent`), supporting scopes and optional values.
*   **CLI**: Typer-based interface.
*   **Database Transactions**: Managed by `WorldScheduler`.

## Project Structure

```
ecs_dam_system/
├── dam/
│   ├── __init__.py
│   ├── cli.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── core/
│   │   ├── conceptual/     # Conceptual assets, variants, pages, tags
│   │   │   ├── base_conceptual_info_component.py
│   │   │   ├── comic_book_concept_component.py
│   │   │   ├── base_variant_info_component.py
│   │   │   ├── comic_book_variant_component.py
│   │   │   ├── page_link.py
│   │   │   ├── tag_concept_component.py
│   │   │   └── entity_tag_link_component.py
│   │   └── ...
│   ├── services/
│   │   ├── __init__.py
│   │   ├── comic_book_service.py
│   │   ├── tag_service.py      # Service for managing tags
│   │   └── ...
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── test_comic_book_service.py
│   ├── test_tag_service.py # Tests for tagging
│   └── ...
└── ... (other project files)
```

## Setup Instructions

1.  **Clone, create venv, activate.**
2.  **Install dependencies:** `pip install -e ."[dev,image,ui]"` (or `uv pip install ...`)
3.  **Set up `.env`** from `.env.example`.
4.  **Initialize database:** `dam-cli setup-db` (Alembic paused).

## Usage

**General help:** `dam-cli --help`

### Asset Versioning, Structure, and Grouping (Example: Comic Books)

Manages different versions of a conceptual work and ordered content like pages.

*   **Abstract Bases:** `BaseConceptualInfoComponent`, `BaseVariantInfoComponent`.
*   **Comic Book Specifics:**
    *   `ComicBookConceptComponent`: Defines a comic concept (title, series, issue, year). Attached to a concept entity.
    *   `ComicBookVariantComponent`: Defines a version of a comic concept (language, format). Attached to a file entity, links to the concept.
    *   `PageLink`: Association table linking a comic variant entity to page image entities with ordering.
*   **Services (`dam.services.comic_book_service`):**
    *   Functions for managing concepts, variants, and their pages (create, link, get, set primary, unlink, assign/remove/order pages).

### Tagging System

Tags are defined as conceptual assets themselves and can be applied to any entity.

*   **`TagConceptComponent`**: Defines a tag (name, scope, description, if it allows values). Inherits from `BaseConceptualInfoComponent` and is attached to a dedicated "tag concept entity."
    *   **Scopes:**
        *   `GLOBAL`: Usable on any entity.
        *   `COMPONENT_CLASS_REQUIRED`: Usable only if the target entity has a specific component (e.g., tag "CoverArt" requires `ImagePropertiesComponent`). `tag_scope_detail` stores the required component class name.
        *   `CONCEPTUAL_ASSET_LOCAL`: Usable only on a specific conceptual asset entity (defined in `tag_scope_detail` by its Entity ID) or its variants.
*   **`EntityTagLinkComponent`**: Applies a tag to an entity. Attached to the entity being tagged.
    *   Links to the `TagConceptComponent`'s entity.
    *   Can store an optional `tag_value` if the tag concept allows it (e.g., Tag: "Rating", Value: "5 Stars").
*   **Services (`dam.services.tag_service`):**
    *   `create_tag_concept()`: Defines a new tag.
    *   `apply_tag_to_entity()`: Applies a tag to an entity, validating scope.
    *   `get_tags_for_entity()`: Lists tags on an entity.
    *   `get_entities_for_tag()`: Finds entities with a specific tag.
    *   And other functions for managing tag definitions and applications.

### Available CLI Commands

*   **`setup-db`**: Initializes/Resets database schema.
*   **`add-asset <filepath>`**: Adds a file asset.
*   **`find-file-by-hash <hash_value>`**: Finds asset by hash.
*   **`find-similar-images <image_filepath>`**: Finds similar images.
*   **`ui`**: Launches PyQt6 UI.

(Details of asset ingestion and basic CLI commands remain similar; new CLI commands for versioning and tagging would be added based on the new services).

## Development

*   **Tests:** `uv run pytest`
*   **Lint/Format:** `uv run ruff format .`, `uv run ruff check . --fix`
*   **Type Check:** `uv run mypy .`

### Database Migrations (Alembic - Currently Paused)
Use `dam-cli setup-db`.

This README will be updated as the project evolves.
