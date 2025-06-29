## Agent Instructions

This file provides instructions for AI agents working with this codebase.

### General Guidelines

*   **Coding Conventions:** Please follow PEP 8 for Python code.
*   **Testing:**
    *   All new features must be accompanied by unit tests.
    *   Ensure all tests pass before submitting changes.
    *   The general command to run tests is `uv run pytest -x`.
    *   **UI Tests:** For tests involving PyQt6 UI components (typically found in `tests/test_ui_*.py` or similar), they are designed to run headlessly. Use the following command:
        ```bash
        QT_QPA_PLATFORM=offscreen uv run pytest -x
        ```
        The `QT_QPA_PLATFORM=offscreen` environment variable ensures that Qt does not require a windowing system, allowing UI tests to run in CI environments or headless servers. `pytest-qt` and `pytest-mock` are required dev dependencies (included in `.[all]`).
    *   To run tests with coverage locally, use `uv run pytest --cov=dam --cov-report=term-missing` (ensure `pytest-cov` is installed via dev dependencies). For UI tests with coverage: `QT_QPA_PLATFORM=offscreen uv run pytest --cov=dam --cov-report=term-missing`.
    *   **Environment Setup:** It is recommended to use `uv` for managing virtual environments and dependencies.
        *   Install dependencies (including `pytest`, `pytest-cov`, `pytest-qt`, `pytest-mock`): `uv run pip install -e .[all]` (as specified in `pyproject.toml`, includes all optional groups).
    *   **Assertion Guideline:** Tests **must not** make assertions directly on terminal output (e.g., `stdout`, `stderr`) or log messages. Instead, tests should verify the state of the system, database, or return values of functions. UI tests should verify widget states, properties, or mocked interactions.
*   **Commit Messages:** Follow conventional commit message formats.
*   **CI/CD:** The continuous integration pipeline is managed by GitHub Actions, configured in `.github/workflows/ci.yml`.
    *   It uses Python 3.12.
    *   It installs dependencies (including test dependencies) using `uv run pip install -e .[all]`.
    *   It runs tests using a command similar to `uv run pytest --cov=dam --cov-report=term-missing`. For UI tests, it must also set `QT_QPA_PLATFORM=offscreen`.
    *   Any changes to the build or test process should be reflected there and in this document.

### Specific Instructions

*   **Design Specification**: For guidance on models (Components) and Systems architecture (excluding testing guidelines which are now in this document), please consult the [Design Specification](docs/design_specification.md) document.
*   There are no other specific instructions at this time.
