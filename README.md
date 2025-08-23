# Monorepo for dam and domarkx

This repository is a monorepo containing the following packages:

-   `packages/dam`: The ECS Digital Asset Management (DAM) System. See its [README.md](packages/dam/README.md) for more details.
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
- `uv run poe check --package sire`
- `uv run poe test --package sire`

## Syncing Dependencies

When you pull new changes, you may need to update the dependencies. To do so, run:

```sh
uv sync --all-extras
```
