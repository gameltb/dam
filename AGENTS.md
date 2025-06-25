## Agent Instructions

This file provides instructions for AI agents working with this codebase.

### General Guidelines

*   **Coding Conventions:** Please follow PEP 8 for Python code.
*   **Testing:**
    *   All new features must be accompanied by unit tests.
    *   Ensure all tests pass before submitting changes.
    *   The command to run tests is `pytest`.
    *   Dependencies, including test dependencies like `pytest`, are managed with `uv` and specified in `pyproject.toml`. Install them using `uv pip install -e ".[image,dev]"`.
*   **Commit Messages:** Follow conventional commit message formats.
*   **CI/CD:** The continuous integration pipeline is managed by GitHub Actions, configured in `.github/workflows/ci.yml`.
    *   It uses Python 3.12.
    *   It installs dependencies using `uv pip install -e ".[image,dev]"`.
    *   It runs tests using `pytest`.
    *   Any changes to the build or test process should be reflected there and in this document.

### Specific Instructions

*   There are no specific instructions at this time.
