"""Utility functions for the Sire package."""


def human_readable_filesize(sz: int, pref_sz: float | None = None) -> str:
    """
    Convert a size in bytes to a human-readable format.

    Args:
        sz: The size in bytes.
        pref_sz: The preferred size for scaling.

    Returns:
        A human-readable string representation of the size.

    """
    if pref_sz is None:
        pref_sz = float(sz)
    prefixes = ["B  ", "KiB", "MiB", "GiB", "TiB", "PiB"]
    prefix = prefixes[0]
    for new_prefix in prefixes[1:]:
        if pref_sz < 768 * 1024:
            break
        prefix = new_prefix
        sz //= 1024
        pref_sz /= 1024
    return f"{sz:6d} {prefix}"
