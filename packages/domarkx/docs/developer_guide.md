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

1.  **Create a portable tool function.** This function should contain the core logic of your tool. Name the function with a `tool_` prefix and place it in the `packages/domarkx/domarkx/tools/portable_tools` directory. For file operations, place them in the `file_operations` subdirectory.

    ```python
    # packages/domarkx/domarkx/tools/portable_tools/my_tool.py

    def tool_my_tool(param1: str, param2: int) -> str:
        # Tool logic here
        return f"Result: {param1}, {param2}"
    ```

2.  **The tool is automatically registered.** The tool registry will automatically discover any function in the `portable_tools` directory that starts with the `tool_` prefix. The `@tool_handler` decorator is also applied automatically.

3.  **Use your tool.** You can now get your tool from the tool registry using the `get_tool` function.

    ```python
    from domarkx.tools.tool_registry import get_tool

    my_tool = get_tool("tool_my_tool")
    ```

    If you need to use the tool with a `JupyterToolExecutor`, you can get the unwrapped tool and wrap it with the `ToolWrapper`.

    ```python setup-script
    from domarkx.tool_executors.jupyter import JupyterToolExecutor
    from domarkx.tools.tool_registry import get_unwrapped_tool
    from domarkx.tools.tool_wrapper import ToolWrapper

    # Create an executor instance
    my_executor = JupyterToolExecutor()

    # Get the unwrapped tool
    my_tool_unwrapped = get_unwrapped_tool("tool_my_tool")

    # Wrap the tool
    my_tool_wrapped = ToolWrapper(tool_func=my_tool_unwrapped, executor=my_executor)

    # Add the wrapped tool to the list of tools
    tools = [my_tool_wrapped]
    tool_executors = [my_executor]
    ```
