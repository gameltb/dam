"""Handles caching of GGUF file metadata."""

import json
from pathlib import Path
from typing import Any, cast

CACHE_FILE = Path("gguf_metadata_cache.json")


def load_metadata_cache() -> dict[str, Any]:
    """Load the GGUF metadata cache from a file."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))
    except (OSError, json.JSONDecodeError):
        return {}


def save_metadata_cache(cache: dict[str, Any]) -> None:
    """Save the GGUF metadata cache to a file."""
    try:
        with CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass  # Ignore errors saving cache


def get_cached_gguf_metadata(model_path: Path, cache: dict[str, Any]) -> dict[str, Any] | None:
    """
    Get GGUF metadata from cache if it's valid.

    The cache is considered valid if the file's modification time and size
    match the stored values.
    """
    path_str = str(model_path)
    if path_str not in cache:
        return None

    cached_data = cache[path_str]
    try:
        stat = model_path.stat()
        if cached_data.get("st_mtime") == stat.st_mtime and cached_data.get("st_size") == stat.st_size:
            return cached_data.get("metadata")
    except FileNotFoundError:
        pass
    return None


def update_gguf_metadata_cache(model_path: Path, metadata: dict[str, Any], cache: dict[str, Any]) -> None:
    """Update the cache with new GGUF metadata."""
    path_str = str(model_path)
    try:
        stat = model_path.stat()
        cache[path_str] = {
            "st_mtime": stat.st_mtime,
            "st_size": stat.st_size,
            "metadata": metadata,
        }
    except FileNotFoundError:
        pass
