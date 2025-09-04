## Agent Instructions for the `dam` Package

This document provides specific instructions for working on the `dam` package, which is the core of the Digital Asset Management system.

### Package Structure

The `dam` package is organized as follows:

*   `src/dam/`: The main source code directory.
    *   `core/`: Contains the fundamental ECS framework, including the `World`, `Plugin`, and `System` concepts.
    *   `models/`: Defines the data components. These are organized by their functional area (e.g., `conceptual`, `hashes`, `metadata`).
    *   `functions/`: Contains the business logic for interacting with the ECS world and its components.
    *   `systems/`: Contains the ECS systems that operate on entities.
    *   `resources/`: Defines shared resources that can be accessed by systems.

### Development Guidelines

*   **Architectural Preference: Commands over Events/Stages**
    *   When implementing new functionality in the `dam` package or its plugins, **prefer using the Command system**.
    *   Avoid using Events, Component Markers, or Stages for triggering system logic unless the task explicitly requires an event-driven or stage-based workflow. Commands provide a clearer, more direct control flow for most operations.

*   **Adding a new Component:**
    1.  Decide on the appropriate subdirectory in `src/dam/models/`.
    2.  Create a new Python file for the component.
    3.  Define the component as a dataclass, inheriting from `BaseComponent` or another appropriate base class.
    4.  Ensure the component is a `MappedAsDataclass`.
*   **Adding a new System:**
    1.  When adding a new system, first consider if it can be added to an existing plugin package (e.g., `dam_psp`, `dam_semantic`).
    2.  If the new system is not a good fit for an existing plugin, create a new plugin package for it.
    3.  Create a new Python file for the system in the appropriate package's `systems/` directory. The system will be automatically discovered and registered by the framework.

### Testing

*   Unit tests for the `dam` package are located in `packages/dam/tests/`.
*   All new features must be accompanied by unit tests.
*   To run the tests for this package, use the command: `uv run poe test --package dam`
*   To run tests with coverage: `uv run poe test-cov --package dam`

### Assertion Guideline

Tests **must not** make assertions directly on terminal output (e.g., `stdout`, `stderr`) or log messages. Instead, tests should verify the state of the system, database, or return values of functions.
