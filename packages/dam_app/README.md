# DAM App

This package provides the main command-line interface (CLI) for the [Digital Asset Management (DAM)](../dam/README.md) system.

## Features

*   Provides a rich set of commands for managing and interacting with the DAM system.
*   Uses a plugin-based architecture to load optional features, such as `dam_psp` and `dam_semantic`.

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
4.  **Set up your `.env` file** by copying the `.env.example` file and modifying it as needed.
5.  **Initialize the database:**
    ```bash
    dam-cli setup-db
    ```

## Development Setup with Docker Compose

For a more robust and persistent local development setup, example Docker Compose files are provided in the `docker/` directory within this package. They simplify the process of starting the required PostgreSQL database.

**Key benefits:**
- **Persistence:** The provided `docker-compose.yml` uses a Docker volume to store the database data, so your data will persist even if you stop and restart the container.
- **Configuration:** It is pre-configured to work with the application's default settings.
- **Simplicity:** It's easier to manage than long `docker run` commands.

### Usage

1.  **Navigate to the docker directory:**
    ```bash
    cd packages/dam_app/docker
    ```
2.  **Copy `.env.example` to `.env`:**
    ```bash
    cp .env.example .env
    ```
3.  **(Optional) Configure Data Path:** Open the `.env` file and change the `POSTGRES_DATA_PATH` to specify where you want to store the database files on your local machine. If you leave it unset, a `pg_data` directory will be created in the `docker` folder.
4.  **Start the database:**
    ```bash
    docker compose up -d
    ```
5.  **Stop the database:**
    ```bash
    docker compose down
    ```

## Usage

Once installed, the `dam-cli` command will be available in your environment.

**General help:**
```bash
dam-cli --help
```

The CLI is organized into several subcommands. For more information on the available commands and the concepts behind them, please refer to the main [DAM documentation](../dam/README.md).
