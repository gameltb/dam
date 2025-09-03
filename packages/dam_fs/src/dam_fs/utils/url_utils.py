"""
This module provides utility functions for parsing and resolving DAM URLs.
"""

from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dam_fs.functions.file_storage import get_file_path as get_cas_path

from dam.core.config import WorldConfig


def parse_dam_url(url: str) -> dict:
    """
    Parses a DAM URL into its components.
    Format: dam://<storage_type>/<path_to_file_or_archive>[#<path_inside_archive>][?pwd=<pwd1>&pwd2=<pwd2>...]
    """
    parsed = urlparse(url)
    storage_type = parsed.hostname
    path = parsed.path.lstrip("/")
    fragment = parsed.fragment
    query = parse_qs(parsed.query)

    passwords = []
    if "pwd" in query:
        passwords.append(query.pop("pwd")[0])

    i = 2
    while f"pwd{i}" in query:
        passwords.append(query.pop(f"pwd{i}")[0])
        i += 1

    return {
        "storage_type": storage_type,
        "path": path,
        "fragment": fragment,
        "query": query,
        "passwords": passwords,
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
        path_part = parsed.path.lstrip("/")

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
