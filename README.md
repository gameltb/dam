# Monorepo for dam and domarkx

This repository is a monorepo containing the following packages:

-   `packages/dam`: The ECS Digital Asset Management (DAM) System. See its [README.md](packages/dam/README.md) for more details.
-   `packages/domarkx`: A tool for making documentation executable. See its [README.md](packages/domarkx/README.md) for more details.

Please refer to the `AGENTS.md` file for instructions on how to work with this codebase.

## Quick Start

**TL;DR**, run all checks with:

```sh
uv sync --all-extras
source .venv/bin/activate
poe check
```

## Setup

`uv` is a package manager that assists in creating the necessary environment and installing packages to run AutoGen.

- [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/).

To upgrade `uv` to the latest version, run:

```sh
uv self update
```

## Virtual Environment

During development, you may need to test changes made to any of the packages.\
To do so, create a virtual environment where the AutoGen packages are installed based on the current state of the directory.\
Run the following commands at the root level of the Python directory:

```sh
uv sync --all-extras
source .venv/bin/activate
```

- `uv sync --all-extras` will create a `.venv` directory at the current level and install packages from the current directory along with any other dependencies. The `all-extras` flag adds optional dependencies.
- `source .venv/bin/activate` activates the virtual environment.

## Common Tasks

To create a pull request (PR), ensure the following checks are met. You can run each check individually:

- Format: `poe format`
- Lint: `poe lint`
- Test: `poe test` (does not collect coverage)
- Test with coverage: `poe test-cov`
- Mypy: `poe mypy`
- Pyright: `poe pyright`
- Check samples in `python/samples`: `poe samples-code-check`

Alternatively, you can run all the checks with:
- `poe check`

To run checks on a specific package, use the `--package` flag:
- `poe check --package sire`
- `poe test --package sire`

> [!NOTE]
> These need to be run in the virtual environment.

## Syncing Dependencies

When you pull new changes, you may need to update the dependencies.
To do so, first make sure you are in the virtual environment, and then in the `python` directory, run:

```sh
uv sync --all-extras
```

This will update the dependencies in the virtual environment.
