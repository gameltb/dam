## Agent Instructions

This file provides instructions for AI agents working with this codebase.

This is a monorepo containing multiple Python packages under the `packages/` directory. Each package is a separate project with its own dependencies and tests.

### General Guidelines

*   **Coding Conventions:** Please follow PEP 8 for Python code.
*   **Commit Messages:** Follow conventional commit message formats.
*   **Worktree/Subtree structure**: This repo is a monorepo, with two main packages `packages/dam` and `packages/domarkx`.
*   **CI/CD:** The continuous integration pipeline is managed by GitHub Actions, configured in `.github/workflows/ci.yml`. Any changes to the build or test process should be reflected there and in this document.

### Environment and Testing

It is recommended to use `uv` for managing virtual environments and dependencies.

#### Managing Environment-Specific Dependencies (e.g., CPU-only PyTorch)

Different environments may require different dependencies, such as a CPU-only version of PyTorch in environments without a GPU or with limited disk space. You can manage this by creating separate `requirements.txt` files or by using optional dependency groups in `pyproject.toml`.

**Example using `requirements.txt`:**

You can create a `requirements-cpu.txt` file in the `packages/dam` directory with the following content:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
torchaudio
```

Then, when installing dependencies for a CPU-only environment, you would run:

```bash
cd packages/dam
uv pip install -r requirements-cpu.txt
uv pip install -e .[all]
```

This ensures that the CPU-only version of PyTorch is installed before other dependencies.

**Note for Jules:** When running tests in the provided environment, you **must** install the CPU-only version of PyTorch to avoid running out of disk space.

#### Running Tests for a Single Package

To run tests for a single package, navigate to the package's directory and use the following commands:

1.  **Navigate to the package directory:**
    ```bash
    cd packages/<package-name>
    ```
2.  **Install dependencies:**
    ```bash
    uv run pip install -e .[all]
    ```
3.  **Run tests:**
    ```bash
    uv run pytest -x
    ```
    For coverage reports, use:
    ```bash
    uv run pytest --cov=<package-name> --cov-report=term-missing
    ```
    (Replace `<package-name>` with `dam` or `domarkx`).

#### Running All Tests

To run all tests for all packages, you can use the following script from the root of the repository:

```bash
#!/bin/bash
set -e
for dir in packages/*/; do
  if [ -f "$dir/pyproject.toml" ]; then
    echo "Testing in $dir"
    (cd "$dir" && uv venv && uv run pip install -e .[all] && uv run pytest -x)
  fi
done
```

### Specific Package Instructions

*   **`packages/dam`**: For more specific instructions on the `dam` package, please refer to its [AGENTS.md](packages/dam/AGENTS.md).
*   **`packages/domarkx`**: This package does not have any special instructions at this time.

### Linting and Formatting

This repository uses `ruff` for linting and formatting. You can run it from the root of the repository using `uv`:

```bash
# Check for linting errors
uv run ruff check .

# Fix linting errors automatically
uv run ruff check . --fix

# Format the code
uv run ruff format .
```

### Assertion Guideline

Tests **must not** make assertions directly on terminal output (e.g., `stdout`, `stderr`) or log messages. Instead, tests should verify the state of the system, database, or return values of functions. UI tests should verify widget states, properties, or mocked interactions.
