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

## Tool Executors

The `domarkx` system uses Tool Executors to provide a flexible way to run tools in different environments. This allows for the same tool to be executed locally or in a remote Jupyter kernel without changing the tool's implementation.

### Creating a Custom Tool

To create a custom tool that can be used with a Tool Executor, follow these steps:

1.  **Create a "clean" tool function.** This function should contain the core logic of your tool and should not have any `domarkx`-specific dependencies. Place this function in the `packages/domarkx/domarkx/tools/unconstrained` directory.

    ```python
    # packages/domarkx/domarkx/tools/unconstrained/my_tool.py
    def my_tool(param1: str, param2: int) -> str:
        # ... tool logic ...
        return "result"
    ```

2.  **Use the `ToolWrapper` to make the tool available to `domarkx`.** In your `setup-script`, import the `ToolWrapper` and your clean tool function. Then, create an instance of the `ToolWrapper`, passing your tool function and a `JupyterToolExecutor` instance.

    ```python setup-script
    from domarkx.tool_executors.jupyter import JupyterToolExecutor
    from domarkx.tools.unconstrained.my_tool import my_tool
    from domarkx.tools.tool_wrapper import ToolWrapper

    # Create an executor instance
    my_executor = JupyterToolExecutor()

    # Wrap the tool
    my_tool_wrapped = ToolWrapper(tool_func=my_tool, executor=my_executor)

    # Add the wrapped tool to the list of tools
    tools = [my_tool_wrapped]
    tool_executors = [my_executor]
    ```
