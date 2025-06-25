# ECS Digital Asset Management (DAM) System

This project implements a Digital Asset Management system using an Entity-Component-System (ECS) architecture in Python. The core idea is to represent assets (images, videos, audio, documents, etc.) as Entities (simple IDs), which are then described by attaching various Components (data-only dataclasses). Systems (typically service functions or dedicated modules) operate on entities based on the components they possess.

## Key Technologies & Concepts

*   **SQLAlchemy ORM**: Used for database interaction, with `MappedAsDataclass` to define components as Python dataclasses that are also database models.
*   **Alembic**: Manages database schema migrations.
*   **Modularity**: Each component type (e.g., `FileLocationComponent`, `DimensionsComponent`, `ImagePHashComponent`) is defined in its own Python file within `dam/models/`. Service functions (for adding, getting, updating components) are typically co-located in the component's model file.
*   **Specialized Logic**: Services like file storage (`dam/services/file_storage.py`) are in separate modules.
*   **Granular Components**: Assets are broken down into very specific data pieces. For example, an image entity might have `FileLocationComponent`, `FilePropertiesComponent`, `DimensionsComponent`, and multiple image hash components (pHash, aHash, etc.). This provides maximum flexibility.
*   **Content-Addressable Foundation**: The `add_asset` process uses a content hash (SHA256) to identify if the file's content already exists in the system, reusing the existing Entity if so, and only adding new or updated component data.
*   **Fingerprinting for Similarity**: Various perceptual hash and fingerprint components are generated and stored for different media types. This allows for similarity searches (though the linking of similar items is a separate process from ingestion).
*   **CLI**: A Typer-based command-line interface (`dam/cli.py`) provides user interaction for adding assets, querying, etc.
*   **Database Transactions**: CLI commands are responsible for managing SQLAlchemy session commits and rollbacks. Component service functions typically add objects to the session but do not commit themselves.

## Project Structure

```
ecs_dam_system/
├── dam/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI application
│   ├── models/             # SQLAlchemy models (Components)
│   │   ├── __init__.py
│   │   └── ... (e.g., file_location_component.py)
│   ├── services/           # Business logic, e.g., file storage
│   │   ├── __init__.py
│   │   └── file_storage.py
│   └── core/               # Core functionalities, DB session, settings
│       ├── __init__.py
│       ├── config.py       # Application settings (e.g., using Pydantic)
│       └── database.py     # SQLAlchemy engine, session setup
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

*   **`add-asset <filepath>`**: Adds a new asset file to the DAM system.
    *   Calculates content hashes (SHA256, MD5).
    *   For images, videos, and GIFs, extracts and stores width/height dimensions using `ImageDimensionsComponent`.
    *   For images, it also calculates perceptual hashes (pHash, aHash, dHash).
    *   For videos, it's now conceptualized as a combination of visual frames and audio:
        *   Visual aspects (frame count, frame rate, duration) are stored in `FramePropertiesComponent`.
        *   Audio track details (codec, duration, sample rate) are stored in `AudioPropertiesComponent`.
    *   For animated GIFs, frame count and animation details are stored in `FramePropertiesComponent`.
    *   For standalone audio files, metadata (duration, codec, sample rate) is stored in `AudioPropertiesComponent`.
    *   If the content already exists (based on SHA256 hash), it links the new filename to the existing asset.
    ```bash
    dam-cli add-asset /path/to/your/image.jpg
    dam-cli add-asset /path/to/your/video.mp4
    dam-cli add-asset /path/to/your/audio.mp3
    dam-cli add-asset /path/to/your/animated.gif
    dam-cli add-asset /path/to/your/document.pdf
    ```

*   **`find-file-by-hash <hash_value>`**: Finds an asset by its content hash and displays its properties. This includes:
    *   Basic file properties (name, size, MIME type).
    *   Content hashes (MD5, SHA256).
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

*   **`find-similar-images <image_filepath>`**: Finds images similar to the provided image based on perceptual hashes.
    *   `--phash-threshold <int>`: Max Hamming distance for pHash (default: 4).
    *   `--ahash-threshold <int>`: Max Hamming distance for aHash (default: 4).
    *   `--dhash-threshold <int>`: Max Hamming distance for dHash (default: 4).
    ```bash
    dam-cli find-similar-images /path/to/query_image.png

    dam-cli find-similar-images /path/to/query_image.png --phash-threshold 2 --ahash-threshold 2
    ```

*   **`query-assets-placeholder`**: (Placeholder, hidden) For future component-based queries.


## Development

*   **Running tests:**
    ```bash
    pytest
    ```
*   **Linting and Formatting:**
    ```bash
    ruff check .
    ruff format .
    ```
*   **Type Checking:**
    ```bash
    mypy .
    ```
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
