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

### Assertion Guideline

Tests **must not** make assertions directly on terminal output (e.g., `stdout`, `stderr`) or log messages. Instead, tests should verify the state of the system, database, or return values of functions. UI tests should verify widget states, properties, or mocked interactions.
