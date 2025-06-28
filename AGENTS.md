## Agent Instructions

This file provides instructions for AI agents working with this codebase.

### General Guidelines

*   **Coding Conventions:** Please follow PEP 8 for Python code.
*   **Testing:**
    *   All new features must be accompanied by unit tests.
    *   Ensure all tests pass before submitting changes.
    *   The command to run tests is `uv run pytest -x`.
    *   To run tests with coverage locally, use `uv run pytest --cov=dam --cov-report=term-missing` (ensure `pytest-cov` is installed via dev dependencies).
    *   **Environment Setup:** It is recommended to use `uv` for managing virtual environments and dependencies.
        *   Install dependencies (including `pytest` and `pytest-cov`): `uv run pip install -e .[all]` (as specified in `pyproject.toml`, includes all optional groups).
*   **Commit Messages:** Follow conventional commit message formats.
*   **CI/CD:** The continuous integration pipeline is managed by GitHub Actions, configured in `.github/workflows/ci.yml`.
    *   It uses Python 3.12.
    *   It installs dependencies (including `pytest-cov`) using `uv run pip install -e .[all]`.
    *   It runs tests using `uv run pytest --cov=dam --cov-report=term-missing` to display a coverage report in the CI logs.
    *   Any changes to the build or test process should be reflected there and in this document.

### Specific Instructions

*   There are no specific instructions at this time.
