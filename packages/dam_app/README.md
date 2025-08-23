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

## Usage

Once installed, the `dam-cli` command will be available in your environment.

**General help:**
```bash
dam-cli --help
```

The CLI is organized into several subcommands. For more information on the available commands and the concepts behind them, please refer to the main [DAM documentation](../dam/README.md).
