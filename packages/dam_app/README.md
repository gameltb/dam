# DAM App

This package provides the main command-line interface (CLI) for the [Digital Asset Management (DAM)](../dam/README.md) system.

## Features

*   Provides a rich set of commands for managing and interacting with the DAM system.
*   Uses a plugin-based architecture to load optional features, such as the `dam_psp` plugin.

## Installation

This package is the main entry point for the DAM system. To install it with all its dependencies, run:

```bash
uv pip install -e .
```

To include optional plugins, specify them as extras. For example, to include the PSP plugin:

```bash
uv pip install -e '.[psp]'
```

## Usage

Once installed, the `dam-cli` command will be available in your environment.

**General help:**
```bash
dam-cli --help
```

For more information on the available commands, please refer to the main [DAM documentation](../dam/README.md).
