# ECS Digital Asset Management (DAM) System

This project implements a Digital Asset Management system using an Entity-Component-System (ECS) architecture in Python. The core idea is to represent assets (images, videos, audio, documents, etc.) as Entities (simple IDs), which are then described by attaching various Components (data-only dataclasses). Systems (typically service functions or dedicated modules) operate on entities based on the components they possess.

## Key Features

*   **Modular ECS Core:** Built on a Bevy-like plugin system, allowing for extensible functionality.
*   **Content-Addressable Storage (CAS):** Files are stored by their content hash, ensuring data integrity and deduplication.
*   **Rich Metadata and Properties:** Extracts and stores a wide range of metadata (e.g., EXIF) and file properties (e.g., image dimensions, audio duration).
*   **Flexible Tagging System:** A powerful tagging system with support for scopes and values.
*   **Asset Versioning and Structuring:** Manages different versions of a conceptual work and ordered content (e.g., comic book pages).
*   **Transcoding and Evaluation Framework:** Define transcoding profiles and evaluate their results to find the optimal settings.
*   **Character Management:** Define characters and link them to assets.
*   **Multi-faceted Search:**
    *   Exact file search by content hash.
    *   Image similarity search using perceptual hashes.
    *   Semantic search using text embeddings (via the `dam_semantic` plugin).
*   **Plugin-Based Architecture:** The system can be extended with optional plugins, suchas `dam_psp` for PSP ISO ingestion.
*   **Command-Line Interface:** A comprehensive CLI (`dam-cli`) for interacting with the system.

## Project Structure

The project is a monorepo divided into several packages:

*   `dam`: The core framework, providing the ECS building blocks, services, and core plugins.
*   `dam_app`: The main CLI application, which loads and configures plugins.
*   `dam_psp`: An optional plugin for PSP ISO ingestion.
*   `dam_semantic`: An optional plugin for semantic search.

```
ecs_dam_system/
├── packages/
│   ├── dam/
│   │   ├── src/dam/
│   │   │   ├── __init__.py
│   │   │   ├── core/
│   │   │   ├── models/
│   │   │   └── systems/
│   │   └── ...
│   ├── dam_app/
│   │   ├── src/dam_app/
│   │   │   └── cli.py       # Main CLI application
│   │   └── ...
│   └── dam_psp/
│       ├── src/dam_psp/
│       │   └── __init__.py  # Provides PspPlugin
│       └── ...
└── ... (other project files)
```

## Core Concepts

### Entity-Component-System (ECS)

The architecture is based on the ECS pattern, which promotes a data-oriented approach to programming.

*   **Entities:** Simple identifiers for assets.
*   **Components:** Data-only dataclasses that describe the properties of an entity. They are also SQLAlchemy models for database persistence.
*   **Systems:** Logic that operates on entities based on the components they possess.

### Services and Systems

Service and system modules are self-contained and designed to be modular.

*   The `dam.services` and `dam.systems` packages do not expose all modules through their `__init__.py` files to support optional dependencies.
*   The `dam.systems` package dynamically discovers and imports all system modules at runtime.

### Detailed Feature Explanations

#### Asset Versioning, Structure, and Grouping (Example: Comic Books)

Manages different versions of a conceptual work and ordered content like pages.

*   **Abstract Bases:** `BaseConceptualInfoComponent`, `BaseVariantInfoComponent`.
*   **Comic Book Specifics:**
    *   `ComicBookConceptComponent`: Defines a comic concept (title, series, issue, year).
    *   `ComicBookVariantComponent`: Defines a version of a comic concept (language, format).
    *   `PageLink`: Association table for ordered pages.
*   **Services (`dam.services.comic_book_service`):** Functions for managing concepts, variants, and pages.

#### Tagging System

Tags are defined as conceptual assets themselves and can be applied to any entity.

*   **`TagConceptComponent`**: Defines a tag with a name, scope, and description.
*   **`EntityTagLinkComponent`**: Applies a tag to an entity, with an optional value.
*   **Services (`dam.services.tag_service`):** Functions for creating, applying, and querying tags.

#### Transcoding and Evaluation

The system supports defining transcoding profiles and evaluating their results.

*   **`TranscodeProfileComponent`**: Defines a transcoding profile with a tool, parameters, and output format.
*   **`TranscodedVariantComponent`**: Links a transcoded asset to its original and the profile used.
*   **`EvaluationRunComponent` and `EvaluationResultComponent`**: For systematic evaluation of transcoding profiles.

#### Character Management

The system allows defining character concepts and linking them to assets.

*   **`CharacterConceptComponent`**: Defines a character.
*   **`EntityCharacterLinkComponent`**: Links an asset to a character.

## Development

*   **Tests:** `poe test`
*   **Tests with coverage:** `poe test-cov`
*   **Lint/Format:** `poe format`, `poe lint`
*   **Type Check:** `poe mypy`

### Database Migrations (Alembic - Currently Paused)

Database migrations are managed by Alembic but are currently paused. Use `dam-cli setup-db` to initialize the database schema.

This README will be updated as the project evolves.
