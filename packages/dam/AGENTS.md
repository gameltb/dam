## Agent Instructions for the `dam` Package

This document provides specific instructions for working on the `dam` package, which is the core of the Digital Asset Management system.

### Package Structure

The `dam` package is organized as follows:

*   `src/dam/`: The main source code directory.
    *   `core/`: Contains the fundamental ECS framework, including the `World`, `Plugin`, and `System` concepts.
    *   `models/`: Defines the data components. These are organized by their functional area (e.g., `conceptual`, `hashes`, `metadata`).
    *   `services/`: Contains the business logic for interacting with the ECS world and its components.
    *   `systems/`: Contains the ECS systems that operate on entities.
    *   `resources/`: Defines shared resources that can be accessed by systems.

### Development Guidelines

*   **Adding a new Component:**
    1.  Decide on the appropriate subdirectory in `src/dam/models/`.
    2.  Create a new Python file for the component.
    3.  Define the component as a dataclass, inheriting from `BaseComponent` or another appropriate base class.
    4.  Ensure the component is a `MappedAsDataclass`.
*   **Adding a new System:**
    1.  Create a new Python file in `src/dam/systems/`.
    2.  The system will be automatically discovered and registered by the framework.
*   **Testing:**
    *   Unit tests for the `dam` package are located in `packages/dam/tests/`.
    *   All new features must be accompanied by unit tests.
    *   To run the tests for this package, use the command: `pytest packages/dam/tests/`
    *   To run tests with coverage: `pytest --cov=dam packages/dam/tests/`

### Assertion Guideline

Tests **must not** make assertions directly on terminal output (e.g., `stdout`, `stderr`) or log messages. Instead, tests should verify the state of the system, database, or return values of functions.
