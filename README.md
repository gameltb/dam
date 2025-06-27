# ECS Digital Asset Management (DAM) System

This project implements a Digital Asset Management system using an Entity-Component-System (ECS) architecture in Python. The core idea is to represent assets (images, videos, audio, documents, etc.) as Entities (simple IDs), which are then described by attaching various Components (data-only dataclasses). Systems (typically service functions or dedicated modules) operate on entities based on the components they possess.

## Key Technologies & Concepts

*   **SQLAlchemy ORM**: Used for database interaction, with `MappedAsDataclass` to define components as Python dataclasses that are also database models.
*   **Alembic**: Manages database schema migrations.
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
│   │       ├── __init__.py
│   │       ├── add_asset_dialog.py
│   │       ├── find_asset_by_hash_dialog.py
│   │       ├── find_similar_images_dialog.py
│   │       └── world_operations_dialogs.py
│   ├── models/             # SQLAlchemy models (Components)
│   │   ├── __init__.py
│   │   └── ... (e.g., file_location_component.py)
│   ├── services/           # Business logic, e.g., file storage, asset creation helpers
│   │   ├── __init__.py
│   │   └── ... (e.g., asset_service.py, file_storage.py)
│   ├── systems/            # ECS Systems
│   │   ├── __init__.py
│   │   └── metadata_systems.py
│   └── core/               # Core ECS framework, DB session, settings
│       ├── __init__.py
│       ├── components_markers.py
│       ├── config.py       # Application settings (Pydantic)
│       ├── database.py     # SQLAlchemy engine, session setup
│       ├── resources.py    # ResourceManager and Resource definitions
│       ├── stages.py       # SystemStage enum
│       ├── system_params.py # Annotated types for system dependency injection
│       └── systems.py      # @system decorator, WorldScheduler
├── tests/                  # Pytest tests
│   ├── __init__.py
│   └── ...
├── .env.example            # Example environment variables
├── .gitignore
├── alembic.ini             # Alembic configuration
├── alembic/                # Alembic migration scripts
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
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
    python3.12 -m venv venv  # Or python -m venv venv if your default python is 3.12+
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -e .
    ```
    The `-e .` installs the package in editable mode.

    Alternatively, you can use [uv](https://github.com/astral-sh/uv) (a fast Python package installer and resolver):
    ```bash
    # Install uv if you don't have it
    # pip install uv
    uv venv venv # Create virtual environment (if not already done)
    source venv/bin/activate # Activate
    uv pip install -e ".[image]" # Install main + image dependencies
    # uv pip install -e ".[dev]" # For development dependencies
    ```

    For optional features like perceptual image hashing and multimedia metadata extraction:
    - Image Hashing: `pip install -e ".[image]"` or `uv pip install -e ".[image]"`
    - The system uses `hachoir` for basic multimedia metadata extraction, which is included in the main dependencies.
    - **For the PyQt6 User Interface:** `pip install -e ".[ui]"` or `uv pip install -e ".[ui]"`

4.  **Set up environment variables:**
    *   Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Edit `.env` and provide your database URL (e.g., `DATABASE_URL=sqlite:///./dam.db` for a local SQLite database).

5.  **Initialize the database and run migrations:**
    *   Ensure your `alembic.ini` is configured (especially `sqlalchemy.url`).
    *   Initialize Alembic (if not already done, but it should be part of the project structure):
        ```bash
        # alembic init alembic  # Only if 'alembic/' directory doesn't exist
        ```
    *   Generate an initial migration (if starting fresh and models are defined):
        ```bash
        # Example: You might need to edit alembic/env.py to point to your Base metadata
        # from dam.models.base import Base # Assuming you have a Base for models
        # target_metadata = Base.metadata
        # alembic revision -m "Initial schema"
        ```
    *   Apply migrations:
        ```bash
        alembic upgrade head
        ```

## Usage

The primary way to interact with the DAM system is through its Command Line Interface (CLI).

**General help:**
```bash
dam-cli --help
```

### Available Commands

*   **`setup-db`**: Initializes the database and creates all necessary tables. Run this once before using other commands if the database is new.
    ```bash
    dam-cli setup-db
    ```

*   **`add-asset <filepath>`**: Adds a new asset file or references an existing one.
    *   Core operation:
        *   Calculates content hashes (SHA256, MD5).
        *   Creates an `Entity`.
        *   Creates `OriginalSourceInfoComponent` to classify the source (e.g., local file, web, reference) using its `source_type` field. This component no longer stores the original filename or path directly.
        *   Creates `FilePropertiesComponent` to store the original filename, file size, and MIME type.
        *   Creates `FileLocationComponent` to store the path to the asset's content (either within DAM managed storage or the external reference path).
    *   For images, it also calculates and stores perceptual hashes (pHash, aHash, dHash) during this initial step.
    *   Marker for System: Adds a `NeedsMetadataExtractionComponent` to the entity.
    *   Scheduled System: After the `add-asset` command completes the core addition, the `MetadataExtractionSystem` is scheduled to run via `world.execute_stage(SystemStage.METADATA_EXTRACTION)`. This system is responsible for:
        *   Extracting detailed metadata using Hachoir (if available).
        *   Creating `ImageDimensionsComponent` for visual media.
        *   Creating `FramePropertiesComponent` for videos and animated GIFs.
        *   Creating `AudioPropertiesComponent` for audio files and audio tracks in videos.
    *   If the content already exists (based on SHA256 hash), it links the new source information to the existing asset entity and may update/add missing components like perceptual hashes or trigger metadata extraction if needed.
    *   Options:
        *   `--no-copy`: Adds the asset by reference. `FileLocationComponent` will store its original path and `OriginalSourceInfoComponent.source_type` will indicate it's a reference.
        *   `-r, --recursive`: If `<filepath>` is a directory, process files recursively.
    ```bash
    dam-cli add-asset /path/to/your/image.jpg
    dam-cli add-asset /path/to/your/video.mp4
    dam-cli add-asset /path/to/your/audio.mp3
    dam-cli add-asset /path/to/your/animated.gif
    dam-cli add-asset /path/to/your/document.pdf
    ```

*   **`find-file-by-hash <hash_value>`**: Finds an asset by its content hash and displays its properties directly to the console. This includes:
    *   Entity ID.
    *   Basic file properties (original filename, size, MIME type from `FilePropertiesComponent`).
    *   Content hashes (MD5, SHA256 from `ContentHashMD5Component`, `ContentHashSHA256Component`).
    *   File Locations (`FileLocationComponent`): Displays contextual filename (if any), content identifier (hash), the physical path/key (within DAM storage or the external reference path), and storage type (e.g., `local_cas`, `local_reference`) for each location.
    *   Image dimensions (width, height) for visual assets.
    *   Frame properties (frame count, rate, duration) for videos and animated GIFs.
    *   Audio properties (codec, duration, sample rate) for audio files and audio tracks within videos.
    *   Perceptual hashes for images.
    *   Options:
        *   `--hash-type <type>`: Specify the hash type (e.g., `sha256`, `md5`). Defaults to `sha256`.
        *   `--file <filepath>` or `-f <filepath>`: Calculate the hash of the given file and use that for searching.
    ```bash
    # Find by providing SHA256 hash directly
    dam-cli find-file-by-hash "a1b2c3d4..."

    # Find by providing MD5 hash directly
    dam-cli find-file-by-hash "x1y2z3w4..." --hash-type md5

    # Find by calculating SHA256 hash of a file
    dam-cli find-file-by-hash --file /path/to/somefile.txt

    # Find by calculating MD5 hash of a file
    dam-cli find-file-by-hash --file /path/to/somefile.txt --hash-type md5
    ```

*   **`find-similar-images <image_filepath>`**: Finds images similar to the provided image based on perceptual hashes and displays the results to the console. Results include Entity ID, filename, distance, and hash type for each match.
    *   `--phash-threshold <int>`: Max Hamming distance for pHash (default: 4).
    *   `--ahash-threshold <int>`: Max Hamming distance for aHash (default: 4).
    *   `--dhash-threshold <int>`: Max Hamming distance for dHash (default: 4).
    ```bash
    dam-cli find-similar-images /path/to/query_image.png

    dam-cli find-similar-images /path/to/query_image.png --phash-threshold 2 --ahash-threshold 2
    ```

*   **`query-assets-placeholder`**: (Placeholder, hidden) For future component-based queries.

*   **`ui`**: Launches the PyQt6-based graphical user interface for managing assets.
    ```bash
    dam-cli ui
    ```
    Make sure you have installed the UI dependencies: `pip install -e ".[ui]"` or `uv pip install -e ".[ui]"`.

    The UI allows you to:
    *   List assets from the currently selected DAM world.
    *   Search for assets by filename.
    *   Filter assets by their MIME type.
    *   View all components of a selected asset by double-clicking it.
    *   Add new assets (equivalent to `dam-cli add-asset`) via 'File > Add Asset(s)...'.
    *   Find assets by content hash (equivalent to `dam-cli find-file-by-hash`) via 'File > Find Asset by Hash...'.
    *   Find similar images (equivalent to `dam-cli find-similar-images`) via 'File > Find Similar Images...'.
    *   Export the current world's data to a JSON file (equivalent to `dam-cli export-world`) via 'Tools > Export Current World...'.
    *   Import data from a JSON file into the current world (equivalent to `dam-cli import-world`) via 'Tools > Import Data into Current World...'.
    *   Merge another world into the current world (DB-to-DB, equivalent to `dam-cli merge-worlds-db`) via 'Tools > Merge Worlds...'.
    *   Split the current world into two other worlds based on criteria (DB-to-DB, equivalent to `dam-cli split-world-db`) via 'Tools > Split Current World...'.
    *   Set up the database for the current world (equivalent to `dam-cli setup-db`) via 'Tools > Setup Database for Current World...'.
    *   The active world for these operations is determined by the `--world` CLI option, `DAM_CURRENT_WORLD` environment variable, or the default world setting, similar to other CLI commands.


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
*   **System Registration & Execution**:
    When Python imports modules containing system functions (decorated with `@system` or `@listens_for`), metadata about these systems is collected globally. For a system to actually run in a specific `World`, it must be explicitly registered with that `World` instance using `world.register_system(...)`.
    Core systems are registered via `dam.core.world_registrar.register_core_systems()` when worlds are initialized by standard application entry points (like the CLI or tests). Developers adding new systems that should be part of this core set should add them to this registrar. For systems specific to certain operations or worlds, manual registration after world creation is appropriate.
    The `WorldScheduler` then executes these registered systems at defined stages or in response to events.
*   **Logging:** The system uses standard Python logging. Output is to stderr by default. Set the `DAM_LOG_LEVEL` environment variable (e.g., to `DEBUG`) for more detailed logs. See `doc/developer_guide.md` for more details.

### Testing Notes

*   **Creating Test Images for Similarity Search**: When testing image similarity features (e.g., `find-similar-images`), it's important to have test images that are distinct yet have known perceptual hash similarities. The `Pillow` library is useful for this.
    *   To ensure images are treated as different assets by the DAM (i.e., have different SHA256 content hashes and thus different Entity IDs), even minor pixel changes are necessary. Simply copying a file will result in the same content hash.
    *   For perceptual hash testing:
        *   Create a base image.
        *   Create variations by adding subtle changes (e.g., changing a few pixels, adding a small line or dot, slightly altering color).
        *   The `imagehash` library can be used directly to calculate pHash, aHash, and dHash values and their Hamming distances to verify expected similarities before incorporating them into automated tests.
    *   The test suite (`tests/test_cli.py`) includes a helper function `_create_dummy_image` that demonstrates creating simple, small PNG images with controlled variations for testing purposes.

*   **Creating new database migrations (after changing SQLAlchemy models):**
    1.  Ensure your models are imported in a way that Alembic's `env.py` can see their metadata (e.g., via a common `Base` object).
    2.  Generate a new revision:
        ```bash
        alembic revision -m " Descriptive_name_for_migration "
        ```
    3.  Inspect and edit the generated migration script in `alembic/versions/`.
    4.  Apply the migration:
        ```bash
        alembic upgrade head
        ```

This README provides a starting point. It will be updated as the project evolves.
