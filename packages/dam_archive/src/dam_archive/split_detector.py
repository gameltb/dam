"""Provides a function to detect split archive parts."""

import re
from re import Pattern
from typing import NamedTuple


class SplitArchiveInfo(NamedTuple):
    """Information about a detected split archive part."""

    base_name: str
    part_num: int


# List of regex patterns to detect split archives.
# Each pattern must contain two capturing groups:
# 1. The base name of the archive.
# 2. The part number.
# The patterns are ordered from most specific to least specific.
SPLIT_ARCHIVE_PATTERNS: list[Pattern[str]] = [
    # WinRAR-style: myarchive.part1.rar, myarchive.part01.rar
    re.compile(r"^(.*?)\.part(\d+)\.rar$", re.IGNORECASE),
    # 7-Zip style: myarchive.7z.001
    re.compile(r"^(.*?)\.7z\.(\d{3})$", re.IGNORECASE),
    # Old RAR-style: myarchive.r01
    re.compile(r"^(.*?)\.r(\d{2})$", re.IGNORECASE),
    # ZIP style: myarchive.z01
    re.compile(r"^(.*?)\.z(\d{2})$", re.IGNORECASE),
]


def detect(filename: str) -> SplitArchiveInfo | None:
    """
    Detect if a filename belongs to a split archive based on known patterns.

    Args:
        filename: The filename to check.

    Returns:
        A SplitArchiveInfo tuple if it's a split archive part, otherwise None.

    """
    for pattern in SPLIT_ARCHIVE_PATTERNS:
        match = pattern.match(filename)
        if match:
            base_name = match.group(1)
            part_num_str = match.group(2)
            try:
                # For .r00, .r01 style, part number is 0-based, but we want 1-based
                # For all others, it's 1-based. Let's assume most are 1-based.
                # RAR .r00 is technically the first part.
                # A common convention is that .r00 is part 1, .r01 is part 2.
                # However, some tools might treat them as 0-indexed.
                # For simplicity, we will treat the number as the part number.
                part_num = int(part_num_str)
                return SplitArchiveInfo(base_name=base_name, part_num=part_num)
            except ValueError:
                continue
    return None
