# DAM Semantic Plugin

This package is a plugin for the [Digital Asset Management (DAM)](../dam/README.md) system that adds functionality for semantic search.

## Features

*   Generates text embeddings for assets using `sentence-transformers`.
*   Provides a semantic search API to find similar assets based on text queries.

## Installation

This plugin is an optional dependency of the `dam_app` package. To install it, include the `semantic` extra when installing `dam_app`:

```bash
uv pip install -e '.[semantic]'
```

## Usage

Once installed, the `search semantic` command will be available in the `dam-cli`:

```bash
dam-cli search semantic --query "<your query>"
```
