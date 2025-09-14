# ECS Digital Asset Management (DAM) System - Core

This package provides the core framework for the Digital Asset Management (DAM) system, built on an Entity-Component-System (ECS) architecture.

## Key Features

*   **Modular ECS Core:** Built on a Bevy-like plugin system, allowing for extensible functionality.
*   **Content-Addressable Storage (CAS):** Files are stored by their content hash, ensuring data integrity and deduplication.
*   **Flexible Tagging System:** A powerful tagging system with support for scopes and values.
*   **Asset Versioning and Structuring:** Manages different versions of a conceptual work and ordered content (e.g., comic book pages).
*   **Character Management:** Define characters and link them to assets.
*   **Plugin-Based Architecture:** The system is designed to be extended with optional plugins. Media-specific functionality (image, audio, transcode) is provided by separate plugins.

## Project Structure

The project is a monorepo divided into several packages:

*   `dam`: The core framework, providing the ECS building blocks.
*   `dam_app`: The main CLI application, which loads and configures plugins.
*   `dam_media_image`: A plugin for image-related functionality.
*   `dam_media_audio`: A plugin for audio-related functionality.
*   `dam_media_transcode`: A plugin for transcode-related functionality.
*   `dam_psp`: An optional plugin for PSP ISO ingestion.
*   `dam_semantic`: An optional plugin for semantic search.

## Core Concepts

### Entity-Component-System (ECS)

The architecture is based on the ECS pattern, which promotes a data-oriented approach to programming.

*   **Entities:** Simple identifiers for assets.
*   **Components:** Data-only dataclasses that describe the properties of an entity. They are also SQLAlchemy models for database persistence.
*   **Systems:** Logic that operates on entities based on the components they possess.

### Functions and Systems

Function and system modules are self-contained and designed to be modular.

*   The `dam.functions` and `dam.systems` packages do not expose all modules through their `__init__.py` files to support optional dependencies.
*   The `dam.systems` package dynamically discovers and imports all system modules at runtime.

## Development

All development tasks should be run from the root of the repository using `uv run poe ...`.

*   **Tests:** `uv run poe test --package dam`
*   **Tests with coverage:** `uv run poe test-cov --package dam`
*   **Lint/Format:** `uv run poe format --package dam` and `uv run poe lint --package dam`
*   **Type Check:** `uv run poe mypy --package dam`

### Database Migrations (Alembic - Currently Paused)

The DAM system now uses PostgreSQL as the default database for storing asset metadata. For instructions on setting up a local PostgreSQL instance for development, please refer to the main `README.md` at the root of the repository.

Database migrations are managed by Alembic but are currently paused. Use `dam-cli setup-db` to initialize the database schema in your configured database.
