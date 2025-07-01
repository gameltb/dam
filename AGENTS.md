## Agent Instructions

This file provides instructions for AI agents working with this codebase.

### General Guidelines

*   **Coding Conventions:** Please follow PEP 8 for Python code.
*   **Testing:**
    *   All new features must be accompanied by unit tests.
    *   Ensure all tests pass before submitting changes.
    *   The general command to run tests is `uv run pytest -x`.
    *   **UI Tests:** Tests involving PyQt6 UI components are typically found in `tests/test_ui_*.py` or files marked with `@pytest.mark.ui`.
        *   These tests are designed to run headlessly in CI using `QT_QPA_PLATFORM=offscreen`.
        *   **Running UI tests (when supported):** The primary command to run all tests, including UI tests, in an environment that supports it (like CI or a properly configured local setup) is:
            ```bash
            QT_QPA_PLATFORM=offscreen uv run pytest -x
            ```
        *   **Skipping UI tests locally:** UI tests might not run correctly in all local/sandbox environments due to missing display servers or other Qt-specific requirements. To run pytest while skipping these UI tests locally, use the `-k` flag to deselect tests by keyword:
            ```bash
            uv run pytest -x -k "not test_ui_"
            ```
            Alternatively, if tests are consistently marked, you can use `-m "not ui"`.
        *   `pytest-qt` and `pytest-mock` are required dev dependencies for UI testing (included in `.[all]`).
    *   To run tests with coverage locally (excluding UI tests if problematic): `uv run pytest -k "not test_ui_" --cov=dam --cov-report=term-missing`.
    *   For UI tests with coverage (in a suitable environment): `QT_QPA_PLATFORM=offscreen uv run pytest --cov=dam --cov-report=term-missing -m ui`.
    *   **Environment Setup:** It is recommended to use `uv` for managing virtual environments and dependencies.
        *   Install dependencies (including `pytest`, `pytest-cov`, `pytest-qt`, `pytest-mock`): `uv run pip install -e .[all]` (as specified in `pyproject.toml`, includes all optional groups).
        *   **PyTorch Installation Note (for limited disk space environments):** Due to limited disk space in some sandbox environments, PyTorch (a dependency of `sentence-transformers`) should be installed as CPU-only first, before other packages that might pull in a larger GPU-enabled version. Use the following command if encountering space issues:
            ```bash
            uv run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ```
            Then proceed with installing other dependencies like `uv run pip install -e .[all]` or `uv run pip install sentence-transformers`. Note that PyTorch will be CPU-only in such environments.
    *   **Assertion Guideline:** Tests **must not** make assertions directly on terminal output (e.g., `stdout`, `stderr`) or log messages. Instead, tests should verify the state of the system, database, or return values of functions. UI tests should verify widget states, properties, or mocked interactions.
*   **Commit Messages:** Follow conventional commit message formats.
*   **CI/CD:** The continuous integration pipeline is managed by GitHub Actions, configured in `.github/workflows/ci.yml`.
    *   It uses Python 3.12.
    *   It installs dependencies (including test dependencies) using `uv run pip install -e .[all]`.
    *   It runs tests in stages. For example, non-UI tests might run with `uv run pytest -x -k "not test_ui_" --cov=dam --cov-report=term-missing`. UI tests will run in a separate step with `QT_QPA_PLATFORM=offscreen uv run pytest -x -m ui --cov=dam --cov-report=term-missing`.
    *   Any changes to the build or test process should be reflected there and in this document.

### Specific Instructions

*   **Design Specification**: For guidance on models (Components) and Systems architecture (excluding testing guidelines which are now in this document), please consult the [Design Specification](docs/design_specification.md) document.
*   There are no other specific instructions at this time.
