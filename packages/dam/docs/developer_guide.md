# Developer Guide: ECS Digital Asset Management (DAM) System

## 1. Introduction

This document provides guidance for developers working on the ECS Digital Asset Management (DAM) system. This project implements a DAM using an Entity-Component-System (ECS) architecture in Python. The core idea is to represent assets (images, videos, audio, documents, etc.) as Entities (simple IDs), which are then described by attaching various Components (data-only dataclasses). Systems (typically service functions or dedicated modules) operate on entities based on the components they possess.

## 2. Core Architectural Concepts

The system is built upon the Entity-Component-System (ECS) pattern, which promotes flexibility and modularity.

### 2.1. Entities
-   **Definition**: Entities are unique identifiers (typically integers or UUIDs) representing a single digital asset or concept within the system. They don't hold data themselves but act as a central point to which Components are attached.
-   **Implementation**: In our system, Entities are represented by the `dam.models.entity.Entity` SQLAlchemy model, which primarily provides a unique `id`.

### 2.2. Components
-   **Definition**: Components are data-only objects that describe a specific aspect or property of an entity. Each component type defines a specific piece of data.
-   **Implementation**:
    -   Components inherit from `dam.models.base_component.BaseComponent`.
    - Dataclass behavior is inherited from `dam.models.core.base_class.Base`.
    - Components are located in the various `dam_media_*` packages.

### 2.3. BaseComponent
-   Provides common fields: `id`, `entity_id` (FK to `entities.id`), and an `entity` relationship.

### 2.4. Systems
-   **Definition**: Systems encapsulate the logic that operates on entities possessing specific combinations of components.
-   **Implementation**:
    *   Systems are Python functions decorated with `@dam.core.systems.system(stage=SystemStage.SOME_STAGE)`.
    *   They are organized into modules within the `systems/` directory of each package.

### 2.5. Plugins
-   **Definition**: The DAM system is built on a plugin architecture. Each plugin is responsible for registering its own components, systems, and resources.
-   **Implementation**:
    *   Plugins implement the `dam.core.plugin.Plugin` protocol.
    *   The `dam_app` package is responsible for loading plugins.
    *   Plugins can depend on other plugins. The `world.add_plugin()` method prevents duplicate registration.

## 3. Project Structure

A brief overview of the key packages:

*   `dam`: The core framework, providing the ECS building blocks.
*   `dam_app`: The main CLI application, which loads and configures plugins.
*   `dam_media_image`: A plugin for image-related functionality.
*   `dam_media_audio`: A plugin for audio-related functionality.
*   `dam_media_transcode`: A plugin for transcode-related functionality.
*   `dam_psp`: An optional plugin for PSP ISO ingestion.
*   `dam_semantic`: An optional plugin for semantic search.

---

## 4. Guide: Adding a New Component or System

This section walks through the process of adding new functionality to the DAM system.

### 4.1. Guideline for New Systems

When adding a new system, first consider if it can be added to an existing plugin package (e.g., `dam_media_image`, `dam_psp`). If the new system provides functionality that is closely related to an existing plugin, it should be added to that plugin.

If the new system is not a good fit for an existing plugin, create a new plugin package for it. This keeps the codebase modular and allows for optional loading of functionality.

### 4.2. Adding a New Component

The process for adding a new component is as follows:
1.  **Define the Component:** Create a new component class in the appropriate plugin package (e.g., `dam_media_image/models/`).
2.  **Register the Component:** Ensure the component is imported in the `__init__.py` of its package so that SQLAlchemy is aware of it.
3.  **Create a System:** Create a system to operate on the new component.
4.  **Register the System:** Register the system in the plugin's `build` method.

---

## 5. Other Development Aspects

### 5.1. Database Migrations (Alembic Workflow)
-   **Current Status (Important):** Alembic is set up, but its usage for generating and applying migrations is **currently paused**.
-   **Development Database Setup:** For development, use the `dam-cli setup-db` command.

### 5.2. Running Tests

The project uses `pytest` for testing, preferably run via `uv` and `poe`.
-   **Run all tests**:
    ```bash
    uv run poe test
    ```
-   **Test Coverage**:
    ```bash
    uv run poe test-cov
    ```

### 5.3. Code Style and Conventions

-   **Formatting & Linting**: `uv run poe format` and `uv run poe lint`.
-   **Type Checking**: `uv run poe mypy`.
