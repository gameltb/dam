## Agent Instructions

This file provides instructions for AI agents working with this codebase.

This is a monorepo containing multiple Python packages under the `packages/` directory. Each package is a separate project with its own dependencies and tests.

### Quick Start

**TL;DR**, run all checks with:

```sh
uv run poe check
```

### Setup

`uv` is a package manager that assists in creating the necessary environment and installing packages.

- [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/).

To upgrade `uv` to the latest version, run:

```sh
uv self update
```

### Virtual Environment

`uv` automatically manages the virtual environment. To install all dependencies, run:

```sh
uv sync --all-extras
```

This will create a `.venv` directory and install all necessary packages.

#### Managing Environment-Specific Dependencies (e.g., CPU-only PyTorch)

`uv` provides a convenient way to automatically select the correct PyTorch backend for your environment. To install PyTorch, set the `UV_TORCH_BACKEND` environment variable to `auto` and then run the installation command:

```bash
export UV_TORCH_BACKEND=auto
uv pip install torch
```

This will automatically detect the appropriate backend (CUDA, CPU, etc.) and install the correct version of PyTorch. In the CI environment, this is handled automatically.

### Common Tasks

All tasks should be run via `uv run poe ...`. This ensures that the command is executed within the correct virtual environment with all the necessary dependencies.

To create a pull request (PR), ensure the following checks are met. You can run each check individually:

- Format: `uv run poe format`
- Lint: `uv run poe lint`
- Test: `uv run poe test` (does not collect coverage)
- Test with coverage: `uv run poe test-cov`
- Mypy: `uv run poe mypy`
- Pyright: `uv run poe pyright`
- Check samples in `python/samples`: `uv run poe samples-code-check`

Alternatively, you can run all the checks with:
- `uv run poe check`

To run checks on a specific package, use the `--package` flag:
- `uv run poe test --package sire`

#### Running Specific Tests

You can pass arguments directly to `pytest` to run specific tests. To do this, add a `--` separator after the `poe` command, followed by the `pytest` arguments.

For example, to run a specific test file within the `dam_archive` package:

```sh
uv run poe test --package dam_archive -- tests/test_archive_commands.py
```

To run a specific test function within that file:

```sh
uv run poe test --package dam_archive -- tests/test_archive_commands.py::test_bind_split_archive_command_workflow
```

### Syncing Dependencies

When you pull new changes, you may need to update the dependencies. To do so, run:

```sh
uv sync --all-extras
```

### Code Quality

-   **`__init__.py` Re-exports**: Avoid re-exporting symbols from `__init__.py` files. This practice can obscure the actual location of a symbol, making the code harder to navigate and understand. It can also lead to circular import problems and interfere with static analysis tools. Always import symbols directly from the module where they are defined.
-   **Type Checking**: All new or modified code must pass `mypy` and `pyright` checks without introducing new errors. When modifying existing code with type errors, you should aim to fix them as part of your changes. If a fix is not feasible, you may use `# type: ignore` with a clear explanation, but this should be a last resort.
-   **Testing CLI Commands**: When testing CLI commands, avoid using `CliRunner` or similar tools that test the command-line interface directly. Instead, refactor the core logic of the command into a separate, importable function. Unit tests should then be written to test this function directly. This approach leads to more robust, maintainable, and less brittle tests.

### Assertion Guideline

Tests **must not** make assertions directly on terminal output (e.g., `stdout`, `stderr`) or log messages. Instead, tests should verify the state of the system, database, or return values of functions. UI tests should verify widget states, properties, or mocked interactions.

### Specific Package Instructions

*   **`packages/dam`**: For more specific instructions on the `dam` package, please refer to its [AGENTS.md](packages/dam/AGENTS.md).
*   **`packages/domarkx`**: This package does not have any special instructions at this time.
