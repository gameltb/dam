# ECS Digital Asset Management (DAM) System

This project implements a Digital Asset Management system using an Entity-Component-System (ECS) architecture in Python. The core idea is to represent assets (images, videos, audio, documents, etc.) as Entities (simple IDs), which are then described by attaching various Components (data-only dataclasses). Systems (typically service functions or dedicated modules) operate on entities based on the components they possess.

## Key Technologies & Concepts

*   **SQLAlchemy ORM**: Used for database interaction, with `MappedAsDataclass` to define components as Python dataclasses that are also database models.
*   **Alembic**: Manages database schema migrations (currently paused during active schema refactoring).
*   **ECS Core Framework**:
    *   **Entities, Components, Systems**: Standard ECS pattern.
    *   **WorldScheduler**: Manages system execution.
    *   **Dependency Injection**: For systems.
    *   **Resources**: Shared utilities.
*   **Modularity**: Components in `dam/models/`, services in `dam/services/`, systems in `dam/systems/`.
*   **Content-Addressable Storage (CAS)**: Files stored by hash.
*   **Metadata Extraction**: For various file types.
*   **Asset Versioning & Structure**: Flexible model for grouping versions of a conceptual work and defining ordered content (e.g., comic pages). Uses:
    *   Abstract bases: `BaseConceptualInfoComponent`, `BaseVariantInfoComponent`.
    *   Concrete components: e.g., `ComicBookConceptComponent`, `ComicBookVariantComponent`.
    *   Association objects: e.g., `PageLink` for ordered pages.
*   **Tagging System**: Tags are defined as conceptual assets (`TagConceptComponent`) and applied to entities via a link component (`EntityTagLinkComponent`), supporting scopes and optional values.
*   **Transcoding Framework**:
    *   Define reusable transcoding profiles (`TranscodeProfileComponent`) as conceptual assets, specifying tool (e.g., `ffmpeg`, `cjxl`), parameters, and output format.
    *   Transcoded files are ingested as new assets, linked to the original via `TranscodedVariantComponent`.
    *   CLI for managing profiles and applying them to assets.
*   **Evaluation Framework**:
    *   Define evaluation runs (`EvaluationRunComponent`) to test multiple profiles on multiple assets.
    *   Results, including file size and placeholder quality metrics, are stored in `EvaluationResultComponent`.
    *   CLI for creating runs, executing them, and reporting results.
*   **CLI**: Typer-based interface.
*   **Database Transactions**: Managed by `WorldScheduler`.

## Project Structure

```
ecs_dam_system/
├── dam/
│   ├── __init__.py
│   ├── cli.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── core/
│   │   ├── conceptual/     # Conceptual assets, variants, pages, tags
│   │   │   ├── base_conceptual_info_component.py
│   │   │   ├── comic_book_concept_component.py
│   │   │   ├── base_variant_info_component.py
│   │   │   ├── comic_book_variant_component.py
│   │   │   ├── page_link.py
│   │   │   ├── tag_concept_component.py
│   │   │   └── entity_tag_link_component.py
│   │   └── ...
│   ├── services/
│   │   ├── __init__.py
│   │   ├── comic_book_service.py
│   │   ├── tag_service.py      # Service for managing tags
│   │   └── ...
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── test_comic_book_service.py
│   ├── test_tag_service.py # Tests for tagging
│   └── ...
└── ... (other project files)
```

## Setup Instructions

1.  **Clone, create venv, activate.**
2.  **Install dependencies:** `pip install -e ."[dev,image,ui]"` (or `uv pip install ...`)
3.  **Set up `.env`** from `.env.example`.
4.  **Initialize database:** `dam-cli setup-db` (Alembic paused).

## Usage

**General help:** `dam-cli --help`

### Asset Versioning, Structure, and Grouping (Example: Comic Books)

Manages different versions of a conceptual work and ordered content like pages.

*   **Abstract Bases:** `BaseConceptualInfoComponent`, `BaseVariantInfoComponent`.
*   **Comic Book Specifics:**
    *   `ComicBookConceptComponent`: Defines a comic concept (title, series, issue, year). Attached to a concept entity.
    *   `ComicBookVariantComponent`: Defines a version of a comic concept (language, format). Attached to a file entity, links to the concept.
    *   `PageLink`: Association table linking a comic variant entity to page image entities with ordering.
*   **Services (`dam.services.comic_book_service`):**
    *   Functions for managing concepts, variants, and their pages (create, link, get, set primary, unlink, assign/remove/order pages).

### Tagging System

Tags are defined as conceptual assets themselves and can be applied to any entity.

*   **`TagConceptComponent`**: Defines a tag (name, scope, description, if it allows values). Inherits from `BaseConceptualInfoComponent` and is attached to a dedicated "tag concept entity."
    *   **Scopes:**
        *   `GLOBAL`: Usable on any entity.
        *   `COMPONENT_CLASS_REQUIRED`: Usable only if the target entity has a specific component (e.g., tag "CoverArt" requires `ImagePropertiesComponent`). `tag_scope_detail` stores the required component class name.
        *   `CONCEPTUAL_ASSET_LOCAL`: Usable only on a specific conceptual asset entity (defined in `tag_scope_detail` by its Entity ID) or its variants.
*   **`EntityTagLinkComponent`**: Applies a tag to an entity. Attached to the entity being tagged.
    *   Links to the `TagConceptComponent`'s entity.
    *   Can store an optional `tag_value` if the tag concept allows it (e.g., Tag: "Rating", Value: "5 Stars").
*   **Services (`dam.services.tag_service`):**
    *   `create_tag_concept()`: Defines a new tag.
    *   `apply_tag_to_entity()`: Applies a tag to an entity, validating scope.
    *   `get_tags_for_entity()`: Lists tags on an entity.
    *   `get_entities_for_tag()`: Finds entities with a specific tag.
    *   And other functions for managing tag definitions and applications.

### Transcoding and Evaluation

The system supports defining transcoding profiles and evaluating their results.

*   **Transcoding Profiles (`TranscodeProfileComponent`)**:
    *   Define a named profile specifying the command-line `tool` (e.g., `ffmpeg`, `cjxl`, `avifenc`), the `parameters` for the tool (using `{input}` and `{output}` placeholders), and the target `output_format` (e.g., `avif`, `jxl`, `mp4`).
    *   These profiles are conceptual assets themselves.
    *   Created via: `dam-cli transcode profile-create --name <profile_name> --tool <tool> --params "<parameters>" --format <format> [--desc <description>]`
*   **Applying Transcodes (`TranscodedVariantComponent`)**:
    *   When a profile is applied to an asset, the specified tool is executed.
    *   The output is ingested as a new asset.
    *   A `TranscodedVariantComponent` is attached to this new asset, linking it to the original asset and the `TranscodeProfileComponent` used. It also stores the transcoded file size.
    *   Applied via: `dam-cli transcode apply --asset <asset_id_or_hash> --profile <profile_id_or_name>`
*   **Evaluation Runs (`EvaluationRunComponent`)**:
    *   Define a named evaluation run to systematically transcode multiple source assets with multiple transcoding profiles.
    *   Created via: `dam-cli evaluate run-create --name <run_name> [--desc <description>]`
*   **Evaluation Results (`EvaluationResultComponent`)**:
    *   For each transcoding operation performed during an evaluation run, an `EvaluationResultComponent` is created and attached to the transcoded asset.
    *   It stores links to the evaluation run, original asset, profile used, file size, and placeholders for quality metrics (e.g., VMAF, SSIM, PSNR - actual calculation is currently a placeholder).
    *   Evaluation executed via: `dam-cli evaluate run-execute --run <run_id_or_name> --assets <asset_ids_or_hashes_comma_sep> --profiles <profile_ids_or_names_comma_sep>`
*   **Reporting**:
    *   View results of an evaluation run: `dam-cli evaluate report --run <run_id_or_name>`

### General CLI Commands

*   **`dam-cli --help`**: Shows all available commands and subcommands.
*   **`dam-cli --world <world_name> <command>`**: Specifies the ECS world to operate on. Can also be set via `DAM_CURRENT_WORLD` env var or a default in settings.
*   **`dam-cli list-worlds`**: Lists configured ECS worlds.
*   **`dam-cli setup-db`**: Initializes/Resets database schema for the selected world.
*   **`dam-cli add-asset <filepath_or_dir>`**: Adds new asset file(s).
    *   Options: `--no-copy` (add by reference), `-r, --recursive` (process directory recursively).
*   **`dam-cli find-file-by-hash <hash_value_or_path>`**: Finds asset by content hash.
    *   Options: `--hash-type <type>` (md5, sha256), `--file <path>` (calculate hash from file).
*   **`dam-cli find-similar-images <image_filepath>`**: Finds similar images using perceptual hashes.
*   **`dam-cli export-world <filepath.json>`**: Exports world data to JSON.
*   **`dam-cli import-world <filepath.json>`**: Imports world data from JSON.
    *   Option: `--merge` (merge with existing data).
*   **`dam-cli ui`**: Launches the PyQt6 UI (if UI dependencies are installed).

(Details of asset ingestion, versioning, and tagging remain as described previously.)

## Development

*   **Tests:** `uv run pytest`
*   **Lint/Format:** `uv run ruff format .`, `uv run ruff check . --fix`
*   **Type Check:** `uv run mypy .`

### Database Migrations (Alembic - Currently Paused)
Use `dam-cli setup-db`.

This README will be updated as the project evolves.
