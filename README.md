# ECS Digital Asset Management (DAM) System

This project implements a Digital Asset Management system using an Entity-Component-System (ECS) architecture in Python. The core idea is to represent assets (images, videos, audio, documents, etc.) as Entities (simple IDs), which are then described by attaching various Components (data-only dataclasses). Systems (typically service functions or dedicated modules) operate on entities based on the components they possess.

## Key Technologies & Concepts

*   **SQLAlchemy ORM**: Used for database interaction, with `MappedAsDataclass` to define components as Python dataclasses that are also database models.
*   **Alembic**: Manages database schema migrations (currently paused during active schema refactoring).
*   **ECS Core Framework**:
    *   **Systems**: Logic that operates on entities with specific components. Implemented as functions decorated with `@system`, supporting asynchronous execution and stage-based processing (e.g., `METADATA_EXTRACTION`).
    *   **WorldScheduler**: Manages the execution of systems in defined stages or in response to events (event handling is planned).
    *   **Dependency Injection**: Systems declare their dependencies (like `WorldSession`, `WorldConfig`, `WorldContext`, `Resource[T]`, `MarkedEntityList[MarkerComponent]`) using `typing.Annotated` or direct type hints for common types like `WorldContext`.
    *   **Resources**: Shared services or utilities (e.g., `FileOperationsResource`) managed by a `ResourceManager` and injectable into systems.
    *   **Marker Components**: Special components (e.g., `NeedsMetadataExtractionComponent`) used to flag entities for processing by specific systems.
    *   **Querying**: The `ecs_service` provides helper functions for common entity queries, such as finding entities by component attributes or combinations of components.
*   **Modularity**: Components are defined in `dam/models/`. Systems are organized in `dam/systems/`.
*   **Granular Components**: Assets are described by fine-grained data pieces (e.g., `FileLocationComponent`, `ImageDimensionsComponent`, hash components).
*   **Content-Addressable Storage (CAS)**: Files are stored based on their content hash (SHA256), promoting de-duplication.
*   **Metadata Extraction**: Perceptual hashes, dimensions, and other metadata are extracted by dedicated systems after initial asset ingestion.
*   **Asset Versioning & Structure**: A flexible model allows grouping different versions of a conceptual work and defining ordered content (like pages in a comic). This uses:
    *   Abstract base components (`BaseConceptualInfoComponent`, `BaseVariantInfoComponent`).
    *   Concrete implementations (e.g., `ComicBookConceptComponent`, `ComicBookVariantComponent`).
    *   Association objects (e.g., `PageLink`) for ordered many-to-many relationships (like comic pages).
    *   See "Asset Versioning, Structure, and Grouping" under Usage.
*   **CLI**: A Typer-based command-line interface (`dam/cli.py`) for user interaction.
*   **Database Transactions**: Managed by the `WorldScheduler`.

## Project Structure

```
ecs_dam_system/
├── dam/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI application
│   ├── ui/                 # PyQt6 UI code (Optional)
│   ├── models/             # SQLAlchemy models (Components & Association Objects)
│   │   ├── __init__.py
│   │   ├── core/
│   │   ├── conceptual/     # Models for conceptual assets, variants, and page links
│   │   │   ├── base_conceptual_info_component.py
│   │   │   ├── comic_book_concept_component.py
│   │   │   ├── base_variant_info_component.py
│   │   │   ├── comic_book_variant_component.py
│   │   │   └── page_link.py
│   │   └── ...
│   ├── services/           # Business logic
│   │   ├── __init__.py
│   │   ├── comic_book_service.py # Service for managing comic book concepts, variants, and pages
│   │   └── ...
│   ├── systems/            # ECS Systems
│   └── core/               # Core ECS framework, DB session, settings
├── tests/                  # Pytest tests
│   ├── __init__.py
│   ├── test_comic_book_service.py
│   └── ...
├── .env.example
├── .gitignore
├── alembic.ini             # Alembic configuration
├── alembic/                # Alembic migration scripts (currently empty)
│   └── versions/           # Migration scripts (currently empty)
├── pyproject.toml
└── README.md
```

## Setup Instructions

1.  **Clone the repository.**
2.  **Create and activate a virtual environment** (Python 3.12+ recommended).
3.  **Install dependencies:**
    ```bash
    pip install -e ."[dev,image,ui]"
    # Or using uv:
    # uv pip install -e ".[dev,image,ui]"
    ```
4.  **Set up environment variables** (copy `.env.example` to `.env` and edit).
5.  **Initialize the database:**
    ```bash
    dam-cli setup-db
    ```
    (Alembic migrations are currently paused; `setup-db` creates tables from models).

## Usage

**General help:** `dam-cli --help`

### Asset Versioning, Structure, and Grouping (Example: Comic Books)

The system models conceptual works, their variants, and ordered content like pages.

*   **Abstract Bases:**
    *   `BaseConceptualInfoComponent`: For entities representing a work's concept.
    *   `BaseVariantInfoComponent`: For file entities that are variants of a concept, linking to the concept's entity.

*   **Comic Book Specifics:**
    *   **`ComicBookConceptComponent`**: Defines a comic concept (e.g., "Amazing Spider-Man #1, 1963") with fields like `comic_title`, `series_title`, `issue_number`, `publication_year`. This is attached to a dedicated "concept entity."
    *   **`ComicBookVariantComponent`**: Defines a specific version of a comic concept (e.g., an English PDF scan). Attached to a file entity. Fields: `language`, `format`, `is_primary_variant`, etc. Links to the concept entity.
    *   **`PageLink` (Association Object Table):** Links a "comic book variant entity" (the owner) to "page image entities" in an ordered sequence.
        *   Stores: `owner_entity_id` (points to the variant entity), `page_image_entity_id`, `page_number`.
        *   This allows an image to be a page in multiple comic variants (or other owning entities) and supports bidirectional queries (pages for a variant, variants an image is part of).

**Managing Comic Books (via `dam.services.comic_book_service`):**

*   **Concepts & Variants:**
    *   `create_comic_book_concept()`
    *   `link_comic_variant_to_concept()`
    *   `get_variants_for_comic_concept()`
    *   And other related functions for finding, setting primary, unlinking.
*   **Pages for Comic Variants:**
    *   `assign_page_to_comic_variant()`: Adds an image entity as a page at a specific number to a comic variant.
    *   `remove_page_from_comic_variant()`: Removes a specific image from a variant's pages.
    *   `remove_page_at_number_from_comic_variant()`: Removes a page by its number.
    *   `get_ordered_pages_for_comic_variant()`: Retrieves the list of page image entities for a variant, in order.
    *   `get_comic_variants_containing_image_as_page()`: Finds which comic variants use a given image as a page.
    *   `update_page_order_for_comic_variant()`: Replaces the entire page sequence for a variant.

Initial file ingestion via `dam-cli add-asset` creates file entities. Subsequent operations using the service layer (typically via more advanced CLI commands or UI actions not yet built) would create comic concepts, link variants, and manage pages.

### Available CLI Commands

*   **`setup-db`**: Initializes/Resets database schema from models.
    ```bash
    dam-cli setup-db
    ```
*   **`add-asset <filepath>`**: Adds a file asset.
    ```bash
    dam-cli add-asset /path/to/your/image.jpg
    ```
*   **`find-file-by-hash <hash_value>`**: Finds asset by content hash.
*   **`find-similar-images <image_filepath>`**: Finds similar images.
*   **`ui`**: Launches the (optional) PyQt6 UI.

(Refer to previous sections or `--help` for more details on these commands)

## Development

*   **Running tests:** `uv run pytest`
*   **Linting/Formatting:** `uv run ruff format .`, `uv run ruff check . --fix`
*   **Type Checking:** `uv run mypy .`

### Database Migrations (Alembic - Currently Paused)
Alembic is set up but paused. Use `dam-cli setup-db` for schema creation during development. Future re-activation of Alembic will involve generating a baseline from current models.

This README will be updated as the project evolves.
