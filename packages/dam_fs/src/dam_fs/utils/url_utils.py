"""
This module provides utility functions for parsing and resolving file URLs.
"""

import os
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlparse


def get_local_path_for_url(url: str) -> Path:
    """
    Resolves a URL to a local filesystem Path.
    Supports 'file://' scheme.
    """
    if not url:
        raise ValueError("URL cannot be empty.")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower() if parsed.scheme else ""

    if scheme == "file":
        # Standard file URI, e.g., file:///path/to/file
        return Path(os.fsdecode(unquote_to_bytes(parsed.path)))
    else:
        raise ValueError(f"Unsupported URL scheme for local access: '{scheme}://'")
