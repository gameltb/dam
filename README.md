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
    *   **Performance**: Internal optimizations have been implemented for efficient handling of entity processing based on marker components.
*   **Modularity**: Components are defined in `dam/models/`. Systems are organized in `dam/systems/`.
*   **Granular Components**: Assets are described by fine-grained data pieces (e.g., `FileLocationComponent`, `ImageDimensionsComponent`, hash components).
*   **Content-Addressable Storage (CAS)**: Files are stored based on their content hash (SHA256), promoting de-duplication.
*   **Metadata Extraction**: Perceptual hashes, dimensions, and other metadata are extracted by dedicated systems after initial asset ingestion.
*   **Asset Versioning**: A flexible model allows grouping different versions or manifestations of the same conceptual work. This uses abstract base components (`BaseConceptualInfoComponent`, `BaseVariantInfoComponent`) and concrete implementations (e.g., `ComicBookConceptComponent`, `ComicBookVariantComponent`). See "Asset Versioning and Grouping" under Usage.
*   **CLI**: A Typer-based command-line interface (`dam/cli.py`) for user interaction.
*   **Database Transactions**: Managed by the `WorldScheduler` per stage or event dispatch. Failures within these operations now raise specific exceptions (`StageExecutionError`, `EventHandlingError`) for clearer error reporting.

## Project Structure

```
ecs_dam_system/
├── dam/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI application
│   ├── ui/                 # PyQt6 UI code
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   └── dialogs/        # Dialog windows for UI operations
│   ├── models/             # SQLAlchemy models (Components)
│   │   ├── __init__.py
│   │   ├── core/
│   │   ├── conceptual/     # Models for conceptual assets and variants
│   │   │   ├── base_conceptual_info_component.py
│   │   │   ├── comic_book_concept_component.py
│   │   │   ├── base_variant_info_component.py
│   │   │   └── comic_book_variant_component.py
│   │   └── ... (e.g., file_location_component.py)
│   ├── services/           # Business logic
│   │   ├── __init__.py
│   │   ├── comic_book_service.py # Service for managing comic book versions
│   │   └── ...
│   ├── systems/            # ECS Systems
│   │   ├── __init__.py
│   │   └── metadata_systems.py
│   └── core/               # Core ECS framework, DB session, settings
│       ├── __init__.py
│       └── ...
├── tests/                  # Pytest tests
│   ├── __init__.py
│   ├── test_comic_book_service.py
│   └── ...
├── .env.example            # Example environment variables
├── .gitignore
├── alembic.ini             # Alembic configuration
├── alembic/                # Alembic migration scripts (currently empty)
│   ├── env.py
│   ├── script.py.mako
│   └── versions/           # Migration scripts (currently empty)
├── pyproject.toml
└── README.md
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ecs_dam_system
    ```

2.  **Create and activate a virtual environment (Python 3.12+ recommended):**
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -e ."[dev,image,ui]"
    # Or using uv:
    # uv pip install -e ".[dev,image,ui]"
    ```
    This installs the package in editable mode along with development, image processing, and UI extras.

4.  **Set up environment variables:**
    *   Copy `.env.example` to `.env`.
    *   Edit `.env` (e.g., `DATABASE_URL=sqlite:///./dam.db`).

5.  **Initialize the database:**
    *   Since Alembic migrations are currently paused, you'll typically initialize the schema directly from models for development:
    ```bash
    dam-cli setup-db
    ```
    *   (When Alembic is active again): `alembic upgrade head`

## Usage

The primary way to interact with the DAM system is through its Command Line Interface (CLI).

**General help:**
```bash
dam-cli --help
```

### Asset Versioning and Grouping (Example: Comic Books)

The system allows managing different versions or manifestations of a conceptual work. This is achieved through a system of abstract base components and concrete, domain-specific components.

*   **Base Components (Abstract):**
    *   `BaseConceptualInfoComponent`: An abstract marker for entities representing a concept (e.g., the idea of a specific comic book issue).
    *   `BaseVariantInfoComponent`: An abstract marker for entities representing a specific file variant of a concept. It links to the conceptual entity.

*   **Concrete Components (Example for Comic Books):**
    *   **`ComicBookConceptComponent`**: Inherits from `BaseConceptualInfoComponent`. Attached to an `Entity` representing the abstract concept of a comic.
        *   Fields: `comic_title`, `series_title`, `issue_number`, `publication_year`.
    *   **`ComicBookVariantComponent`**: Inherits from `BaseVariantInfoComponent`. Attached to an `Entity` that is an actual file (e.g., a PDF or CBZ of the comic).
        *   Links to the `ComicBookConceptComponent`'s entity via `conceptual_entity_id`.
        *   Fields: `language`, `format`, `scan_quality`, `is_primary_variant`, `variant_description`.

**Managing Comic Book Concepts and Variants:**

The `dam.services.comic_book_service` module provides functions:

*   `create_comic_book_concept()`: Creates an entity representing a comic book concept (e.g., "Amazing Spider-Man #1, 1963").
*   `link_comic_variant_to_concept()`: Links a file entity (e.g., an English PDF scan) to a specific comic book concept entity.
*   `get_variants_for_comic_concept()`: Retrieves all file variants associated with a comic book concept.
*   `get_comic_concept_for_variant()`: Finds the parent comic book concept for a given file variant.
*   `find_comic_book_concepts()`: Searches for comic book concepts.
*   `set_primary_comic_variant()`: Marks one variant as the primary/default for its concept.
*   `unlink_comic_variant()`: Removes the link between a file variant and its concept.

This structure allows for defining other types of conceptual assets (e.g., `MovieConceptComponent`, `MovieVariantComponent`) in the future.

The initial ingestion of a file (`dam-cli add-asset`) creates an entity for that file. Linking it as a variant to a conceptual entity (e.g., a `ComicBookConceptEntity`) is a separate step managed via the domain-specific services (like `comic_book_service`).

### Available Commands

*   **`setup-db`**: Initializes the database and creates all necessary tables based on current models. Run this if the database is new or if models have changed and migrations are not being used.
    ```bash
    dam-cli setup-db
    ```

*   **`add-asset <filepath>`**: Adds a new asset file or references an existing one.
    *   (Details as before, this part of usage remains largely unchanged at the CLI level for initial file ingestion)
    ```bash
    dam-cli add-asset /path/to/your/image.jpg
    ```

*   **`find-file-by-hash <hash_value>`**: (Details as before)
    ```bash
    dam-cli find-file-by-hash "a1b2c3d4..."
    ```

*   **`find-similar-images <image_filepath>`**: (Details as before)
    ```bash
    dam-cli find-similar-images /path/to/query_image.png
    ```

*   **`ui`**: Launches the PyQt6-based graphical user interface.
    ```bash
    dam-cli ui
    ```

## Development

*   **Running tests:**
    ```bash
    uv run pytest
    ```
*   **Linting and Formatting:**
    ```bash
    uv run ruff format .
    uv run ruff check . --fix
    ```
*   **Type Checking:**
    ```bash
    uv run mypy .
    ```
*   **System Registration & Execution**: (Details as before)
*   **Logging:** (Details as before)

### Testing Notes

*   (Details as before)

*   **Database Migrations (Alembic - Currently Paused):**
    While Alembic is set up, its usage for generating and applying migrations is currently paused during this phase of rapid schema evolution. For development, the `dam-cli setup-db` command is used to create tables directly from models. Once the schema stabilizes, Alembic migrations will be re-introduced.
    To re-initialize Alembic in the future:
    1.  Clear the `alembic/versions` directory.
    2.  Ensure `alembic/env.py` points to your `Base.metadata`.
    3.  Generate a new baseline revision: `alembic revision -m "Baseline schema from models" --autogenerate`
    4.  Apply: `alembic upgrade head`

This README provides a starting point. It will be updated as the project evolves.
