# Domarkx Developer Guide

This guide provides instructions for developers working on the `domarkx` codebase.

## Setup Instructions

1.  **Clone the repository.**
2.  **Navigate to the `packages/domarkx` directory.**
3.  **Create a virtual environment and activate it:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```
4.  **Install dependencies:**
    ```bash
    uv pip install -e .[all]
    ```

## Running Tests

To run the test suite, use the following command:

```bash
uv run pytest
```

### Temporary Files

When writing tests that create files, it is recommended to use the `runner.isolated_filesystem()` context manager provided by `typer.testing.CliRunner`. This will create a temporary directory for the test and automatically clean it up afterwards. This is preferred over manual creation and cleanup of temporary directories.

## Coding Conventions

Please follow PEP 8 for Python code. Use `ruff` for linting and formatting:

```bash
# Check for linting errors
uv run ruff check .

# Fix linting errors
uv run ruff check . --fix

# Format code
uv run ruff format .
```

## Project Structure

The `domarkx` project is structured as follows:

```
packages/domarkx/
├── domarkx/
│   ├── __init__.py
│   ├── cli.py          # Command-line interface
│   ├── action/         # Core actions
│   └── ...
├── docs/
│   ├── developer_guide.md
│   └── design_specification.md
├── tests/
│   └── ...
└── pyproject.toml
```
