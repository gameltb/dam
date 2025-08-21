## Agent Instructions

This file provides instructions for AI agents working with this codebase.

This is a monorepo containing multiple Python packages under the `packages/` directory. Each package is a separate project with its own dependencies and tests.

### Quick Start

**TL;DR**, run all checks with:

```sh
uv sync --all-extras
source .venv/bin/activate
poe check
```

### Setup

`uv` is a package manager that assists in creating the necessary environment and installing packages.

- [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/).

To upgrade `uv` to the latest version, run:

```sh
uv self update
```

### Virtual Environment

During development, you may need to test changes made to any of the packages.\
To do so, create a virtual environment where the packages are installed based on the current state of the directory.\
Run the following commands at the root level of the repository:

```sh
uv sync --all-extras
source .venv/bin/activate
```

- `uv sync --all-extras` will create a `.venv` directory at the current level and install packages from the current directory along with any other dependencies. The `all-extras` flag adds optional dependencies.
- `source .venv/bin/activate` activates the virtual environment.

#### Managing Environment-Specific Dependencies (e.g., CPU-only PyTorch)

`uv` provides a convenient way to automatically select the correct PyTorch backend for your environment. To install PyTorch, set the `UV_TORCH_BACKEND` environment variable to `auto` and then run the installation command:

```bash
export UV_TORCH_BACKEND=auto
uv pip install torch
```

This will automatically detect the appropriate backend (CUDA, CPU, etc.) and install the correct version of PyTorch. In the CI environment, this is handled automatically.

### Common Tasks

To create a pull request (PR), ensure the following checks are met. You can run each check individually:

- Format: `poe format`
- Lint: `poe lint`
- Test: `poe test` (does not collect coverage)
- Test with coverage: `poe test-cov`
- Mypy: `poe mypy`
- Pyright: `poe pyright`
- Check samples in `python/samples`: `poe samples-code-check`

Alternatively, you can run all the checks with:
- `poe check`

To run checks on a specific package, use the `--package` flag:
- `poe check --package sire`
- `poe test --package sire`

> [!NOTE]
> These need to be run in the virtual environment.

### Syncing Dependencies

When you pull new changes, you may need to update the dependencies.
To do so, first make sure you are in the virtual environment, and then run:

```sh
uv sync --all-extras
```

This will update the dependencies in the virtual environment.

### Assertion Guideline

Tests **must not** make assertions directly on terminal output (e.g., `stdout`, `stderr`) or log messages. Instead, tests should verify the state of the system, database, or return values of functions. UI tests should verify widget states, properties, or mocked interactions.

### Specific Package Instructions

*   **`packages/dam`**: For more specific instructions on the `dam` package, please refer to its [AGENTS.md](packages/dam/AGENTS.md).
*   **`packages/domarkx`**: This package does not have any special instructions at this time.
