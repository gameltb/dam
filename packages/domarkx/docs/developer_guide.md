# Domarkx Developer Guide

This guide provides instructions for developers working on the `domarkx` codebase.

## Setup Instructions

Please refer to the setup instructions in the root `AGENTS.md` file of this repository for the most up-to-date and consistent setup process. The recommended approach is to use the `uv sync --all-extras` command from the repository root to ensure all dependencies across the monorepo are correctly installed.

## Running Tests

To run the test suite, use the following command:

```bash
uv run pytest
```

### Temporary Files

When writing tests that create files, it is recommended to use the `runner.isolated_filesystem()` context manager provided by `typer.testing.CliRunner`. This will create a temporary directory for the test and automatically clean it up afterwards. This is preferred over manual creation and cleanup of temporary directories.

## Coding Conventions

Please follow PEP 8 for Python code. It is recommended to use `poe` to run all checks, as it is configured for the entire monorepo.

```bash
# Run all checks (format, lint, test, etc.)
poe check

# Or run individual checks
poe format
poe lint
poe test
```

This ensures consistency across the entire project.

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

## Function Execution

`domarkx` allows for the execution of functions (tools) as part of a conversation. This is useful for creating interactive documentation and testing tool integrations.

### Defining Tools

Tools can be defined in a `setup-script` block within your Markdown document. Here are the different ways you can define and use tools:

#### Plain Local Function Call

You can define a simple Python function and add it to the `tools` list.

```python setup-script
from domarkx.models.openrouter import OpenRouterR1OpenAIChatCompletionClient

client = OpenRouterR1OpenAIChatCompletionClient(model="...")

def add(a: int, b: int) -> int:
    return a + b

tools = [add]
```

#### Portable Tools (Local and Remote)

"Portable tools" are designed to be executed in different environments (local or remote).

##### Local Execution

To use a portable tool locally, import it and add it to the `tools` list.

```python setup-script
from domarkx.models.openrouter import OpenRouterR1OpenAIChatCompletionClient
from domarkx.tools.portable_tools.execute_command import tool_execute_command

client = OpenRouterR1OpenAIChatCompletionClient(model="...")

tools = [tool_execute_command]
tool_executors = []
```

##### Remote Execution (Jupyter)

To execute a portable tool remotely, you need a `JupyterToolExecutor` and a `CodeExecutor`. The `ToolFactory` simplifies wrapping the tool.

```python setup-script
from domarkx.models.openrouter import OpenRouterR1OpenAIChatCompletionClient
from domarkx.tools.tool_factory import default_tool_factory
from domarkx.tool_executors.jupyter import JupyterToolExecutor
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

client = OpenRouterR1OpenAIChatCompletionClient(model="...")

# Create a code executor and a Jupyter tool executor
code_executor = LocalCommandLineCodeExecutor()
jupyter_executor = JupyterToolExecutor(code_executor=code_executor)

# Get the unwrapped tool function from the factory
tool_func = default_tool_factory.get_unwrapped_tool("tool_execute_command")

# Wrap the tool with the executor using the factory
tool_execute_command_wrapped = default_tool_factory.wrap_function(func=tool_func, executor=jupyter_executor)

tools = [tool_execute_command_wrapped]
tool_executors = [jupyter_executor]
```

#### Class-Based Tools

You can also define tools as methods of a class.

##### Local Execution

```python setup-script
from domarkx.models.openrouter import OpenRouterR1OpenAIChatCompletionClient

client = OpenRouterR1OpenAIChatCompletionClient(model="...")

class MyTool:
    def add(self, a: int, b: int) -> int:
        return a + b

tools = [MyTool().add]
tool_executors = []
```

##### Remote Execution

```python setup-script
from domarkx.models.openrouter import OpenRouterR1OpenAIChatCompletionClient
from domarkx.tools.tool_factory import default_tool_factory
from domarkx.tool_executors.jupyter import JupyterToolExecutor
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

client = OpenRouterR1OpenAIChatCompletionClient(model="...")

class MyTool:
    def add(self, a: int, b: int) -> int:
        return a + b

# Create executors
code_executor = LocalCommandLineCodeExecutor()
jupyter_executor = JupyterToolExecutor(code_executor=code_executor)

# Wrap the tool method using the factory
my_tool = MyTool()
add_tool_wrapped = default_tool_factory.wrap_function(func=my_tool.add, executor=jupyter_executor)

tools = [add_tool_wrapped]
tool_executors = [jupyter_executor]
```

### Creating Tools from Strings

The `ToolFactory` also allows you to create tools dynamically from a string of Python code. This is useful for scenarios where tools are generated on the fly.

```python setup-script
from domarkx.models.openrouter import OpenRouterR1OpenAIChatCompletionClient
from domarkx.tools.tool_factory import default_tool_factory
from domarkx.tool_executors.jupyter import JupyterToolExecutor
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

client = OpenRouterR1OpenAIChatCompletionClient(model="...")

# Code for the tool as a string
tool_code = """
def multiply(a: int, b: int) -> int:
    return a * b
"""

# Create executors
code_executor = LocalCommandLineCodeExecutor()
jupyter_executor = JupyterToolExecutor(code_executor=code_executor)

# Create the tool from the string
multiply_tool = default_tool_factory.create_tool_from_string(code=tool_code, executor=jupyter_executor)

tools = [multiply_tool]
tool_executors = [jupyter_executor]
```

### Overriding Parameters

You can override parameters in your Markdown document by passing an `override_parameters` dictionary to the `AutoGenSession` constructor. This is useful for dynamically setting values at runtime.

For example, if you have a macro in your Markdown document like this:

```markdown
@[my-variable](domarkx://set?value=default)
```

You can override the value of `my-variable` like this:

```python
from domarkx.autogen_session import AutoGenSession

override_parameters = {"my-variable": {"value": "overridden"}}
```
