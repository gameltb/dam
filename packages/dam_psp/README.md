# DAM PSP Plugin

This package is a plugin for the [Digital Asset Management (DAM)](../dam/README.md) system that adds functionality for ingesting Sony PlayStation Portable (PSP) ISO files.

## Features

*   Extracts metadata from `PARAM.SFO` files within PSP ISOs.
*   Handles ISOs inside `.zip` and `.7z` archives.
*   Creates entities in the DAM system with the extracted metadata and content hashes.

## Installation

This plugin is an optional dependency of the `dam_app` package. To install it, include the `psp` extra when installing `dam_app`:

```bash
uv pip install -e '.[psp]'
```

## Usage

Once installed, the `ingest-psp-isos` command will be available in the `dam-cli`:

```bash
dam-cli ingest-psp-isos <directory>
```
