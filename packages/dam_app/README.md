# DAM App

This package provides the main command-line interface (CLI) for the [Digital Asset Management (DAM)](../dam/README.md) system.

## Features

*   Provides a rich set of commands for managing and interacting with the DAM system.
*   Uses a plugin-based architecture to load optional features, such as `dam_psp` and `dam_semantic`.
*   Supports multiple, isolated "worlds," each with its own database schema and configuration, managed via a central `dam.toml` file.
*   Includes a database migration system powered by Alembic to manage schema changes on a per-world basis.

## Installation

1.  **Clone the repository, create a virtual environment, and activate it.**
2.  **Install the application and its core dependencies:**
    ```bash
    uv pip install -e .
    ```
3.  **To include optional plugins, specify them as extras.** For example:
    *   To include the PSP plugin:
        ```bash
        uv pip install -e '.[psp]'
        ```
    *   To include the semantic search plugin:
        ```bash
        uv pip install -e '.[semantic]'
        ```
    *   To install all plugins and development dependencies:
        ```bash
        uv pip install -e '.[all]'
        ```

## Configuration

The `dam-cli` application is configured using a `dam.toml` file in the root of your project. This file is mandatory and defines the different "worlds" the application can operate on.

### Example `dam.toml`

Here is an example `dam.toml` that defines two worlds: `fs_world` and `archive_world`.

```toml
# Configuration for the DAM application using PostgreSQL

# --- Filesystem World ---
# This world will store its migrations in a custom local directory.
[worlds.fs_world]
[worlds.fs_world.db]
url = "postgresql+psycopg://postgres:postgres@localhost:5432/dam"

[worlds.fs_world.plugins]
names = ["dam-fs", "dam-source"]

[worlds.fs_world.paths]
# Mandatory: The directory to store Alembic migration files for this world.
alembic = "./fs_world_migrations"

# --- Archive World ---
# This world will use a custom storage path for the dam-fs plugin.
[worlds.archive_world]
[worlds.archive_world.db]
url = "postgresql+psycopg://postgres:postgres@localhost:5432/dam"

[worlds.archive_world.plugins]
names = ["dam-fs", "dam-source", "dam-archive"]

[worlds.archive_world.paths]
# Mandatory: The directory to store Alembic migration files for this world.
alembic = "./archive_world_migrations"

[worlds.archive_world.plugin_settings."dam-fs"]
# Optional: A custom storage path for the dam-fs plugin in this world.
storage_path = "./archive_world_storage"
```

## Development Setup with Docker Compose

For a more robust and persistent local development setup, a Docker Compose file is provided in the `docker/` directory to start the required PostgreSQL database.

### Usage

1.  **Navigate to the docker directory:**
    ```bash
    cd packages/dam_app/docker
    ```
2.  **Start the database:**
    ```bash
    docker compose up -d
    ```
3.  **Stop the database:**
    ```bash
    docker compose down
    ```

## Database Migrations

Database schemas are managed on a per-world basis using the `dam-cli db` command, which is a wrapper around Alembic.

**Workflow:**

1.  **Initialize the migration environment for your world.** This only needs to be done once per world. It will create the directory specified in your `dam.toml` and populate it with the necessary Alembic files.
    ```bash
    uv run poe run --package dam_app -- db init --world <your_world_name>
    ```

2.  **Generate a new migration script.** After making changes to your SQLAlchemy models in a plugin, run this command to automatically generate a migration script.
    ```bash
    uv run poe run --package dam_app -- db revision --world <your_world_name> --autogenerate -m "A message describing your changes"
    ```

3.  **Apply the migration to the database.** This will execute the migration script and update your database schema.
    ```bash
    uv run poe run --package dam_app -- db upgrade --world <your_world_name> head
    ```

## Usage

Once installed, all commands should be run via the `uv run poe run --package dam_app` task.

**General help:**
```bash
uv run poe run --package dam_app -- --help
```

The CLI is organized into several subcommands. For more information on the available commands and the concepts behind them, please refer to the main [DAM documentation](../dam/README.md).