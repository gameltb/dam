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

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -e .
    ```
    The `-e .` installs the package in editable mode.

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

**(Placeholder for specific commands once implemented)**
```bash
# Example: Adding an asset
# dam-cli add-asset /path/to/your/image.jpg --tags "photo,nature"

# Example: Querying assets
# dam-cli query --component DimensionsComponent --filter "width>1920"
```

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
