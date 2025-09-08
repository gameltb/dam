from typing import Optional


def human_readable_filesize(sz: int, pref_sz: Optional[float] = None) -> str:
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
