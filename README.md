# Monorepo for dam and domarkx

This repository is a monorepo containing the following packages:

-   `packages/dam`: The core ECS Digital Asset Management (DAM) framework.
-   `packages/dam_app`: The CLI application for the DAM system.
-   `packages/dam_media_image`: A plugin for image-related functionality.
-   `packages/dam_media_audio`: A plugin for audio-related functionality.
-   `packages/dam_media_transcode`: A plugin for transcode-related functionality.
-   `packages/dam_psp`: An optional plugin for PSP ISO ingestion.
-   `packages/dam_semantic`: An optional plugin for semantic search.
-   `packages/domarkx`: A tool for making documentation executable. See its [README.md](packages/domarkx/README.md) for more details.

Please refer to the `AGENTS.md` file for instructions on how to work with this codebase.

## Quick Start

**TL;DR**, run all checks with:

```sh
uv run poe check
```

## Setup

`uv` is a package manager that assists in creating the necessary environment and installing packages.

- [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/).

To upgrade `uv` to the latest version, run:

```sh
uv self update
```

### Database Setup

The DAM system uses a PostgreSQL database to store asset metadata. For local development, the easiest way to get a database running is by using Docker.

Run the following command to start a PostgreSQL container with the `pgvector` extension pre-installed:

```sh
docker run --name dam-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=postgres -e POSTGRES_DB=dam -p 5432:5432 -d pgvector/pgvector:pg17
```

This will start a PostgreSQL 17 server with vector support in the background. The application is configured by default to connect to this database at `postgresql+psycopg://postgres:postgres@localhost:5432/dam`.

To stop the container, run `docker stop dam-postgres`. To start it again, use `docker start dam-postgres`.

## Virtual Environment

`uv` automatically manages the virtual environment. To install all dependencies, run:

```sh
uv sync --all-extras
```

This will create a `.venv` directory and install all necessary packages.

## Common Tasks

All tasks should be run via `uv run poe ...`. This ensures that the command is executed within the correct virtual environment with all the necessary dependencies.

To create a pull request (PR), ensure the following checks are met. You can run each check individually:

- Format: `uv run poe format`
- Lint: `uv run poe lint`
- Test: `uv run poe test` (does not collect coverage)
- Test with coverage: `uv run poe test-cov`
- Mypy: `uv run poe mypy`
- Pyright: `uv run poe pyright`
- Check samples in `python/samples`: `uv run poe samples-code-check`

Alternatively, you can run all the checks with:
- `uv run poe check`

To run checks on a specific package, use the `--package` flag:
- `uv run poe test --package sire`

### Running Specific Tests

You can pass arguments directly to `pytest` to run specific tests. To do this, add a `--` separator after the `poe` command, followed by the `pytest` arguments.

For example, to run a specific test file within the `dam_archive` package:

```sh
uv run poe test --package dam_archive -- tests/test_archive_commands.py
```

To run a specific test function within that file:

```sh
uv run poe test --package dam_archive -- tests/test_archive_commands.py::test_bind_split_archive_command_workflow
```

## Syncing Dependencies

When you pull new changes, you may need to update the dependencies. To do so, run:

```sh
uv sync --all-extras
```
