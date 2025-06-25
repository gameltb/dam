## Agent Instructions

This file provides instructions for AI agents working with this codebase.

### General Guidelines

*   **Coding Conventions:** Please follow PEP 8 for Python code.
*   **Testing:**
    *   All new features must be accompanied by unit tests.
    *   Ensure all tests pass before submitting changes.
    *   The command to run tests is `pytest`.
    *   To run tests with coverage locally, use `pytest --cov=dam --cov-report=term-missing` (ensure `pytest-cov` is installed via dev dependencies).
    *   **Environment Setup:** It is recommended to use `uv` for managing virtual environments and dependencies.
        *   Create a virtual environment: `uv venv .venv` (or your preferred directory name).
        *   Activate the environment: `source .venv/bin/activate` (Linux/macOS) or `.venv\\Scripts\\activate` (Windows).
        *   Install dependencies (including `pytest` and `pytest-cov`): `uv pip install -e ".[image,dev]"` (as specified in `pyproject.toml`).
*   **Commit Messages:** Follow conventional commit message formats.
*   **CI/CD:** The continuous integration pipeline is managed by GitHub Actions, configured in `.github/workflows/ci.yml`.
    *   It uses Python 3.12.
    *   It creates a virtual environment using `uv venv .venv` and activates it.
    *   It installs dependencies (including `pytest-cov`) using `uv pip install -e ".[image,dev]"`.
    *   It runs tests using `pytest --cov=dam --cov-report=term-missing` to display a coverage report in the CI logs.
    *   Any changes to the build or test process should be reflected there and in this document.

### Specific Instructions

*   There are no specific instructions at this time.
