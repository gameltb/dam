"""
This module provides utility functions for parsing and resolving DAM URLs.
"""
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dam.core.config import WorldConfig
from dam.services.file_storage import get_file_path as get_cas_path


def parse_dam_url(url: str) -> dict:
    """
    Parses a DAM URL into its components.
    Format: dam://<storage_type>/<path_to_file_or_archive>[#<path_inside_archive>]
    """
    parsed = urlparse(url)
    storage_type = parsed.hostname
    path = parsed.path.lstrip('/')
    fragment = parsed.fragment
    query = parse_qs(parsed.query)
    return {
        "storage_type": storage_type,
        "path": path,
        "fragment": fragment,
        "query": query,
    }


def get_local_path_for_url(url: str, config: WorldConfig) -> Path:
    """
    Resolves a DAM URL to a local filesystem Path.
    Supports 'dam://local_cas', 'dam://local_reference', and 'file://' schemes.
    """
    if not url:
        raise ValueError("URL cannot be empty.")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower() if parsed.scheme else ""

    if scheme == "dam":
        storage_type = parsed.hostname
        # The path in a dam URL is the content identifier (e.g., hash)
        # or the full path for a reference.
        path_part = parsed.path.lstrip('/')

        if storage_type == "local_cas":
            content_hash = path_part
            cas_path = get_cas_path(content_hash, config)
            if not cas_path:
                raise FileNotFoundError(f"CAS file not found for hash: {content_hash}")
            return cas_path
        elif storage_type == "local_reference":
            # For local references, the path is stored directly.
            return Path(path_part)
        else:
            raise ValueError(f"Unsupported DAM storage type for local access: '{storage_type}'")
    elif scheme == "file":
        # Standard file URI, e.g., file:///path/to/file
        return Path(parsed.path)
    else:
        raise ValueError(f"Unsupported URL scheme for local access: '{scheme}://'")
