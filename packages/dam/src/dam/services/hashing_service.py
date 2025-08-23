"""
This service provides centralized functions for calculating various types of hashes
for files, including content hashes (e.g., MD5, SHA256) and perceptual hashes
for images (e.g., aHash, pHash, dHash).
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, BinaryIO, Dict, List

logger = logging.getLogger(__name__)


def calculate_sha256(filepath: Path) -> str:
    """
    Calculates the SHA256 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        The hex digest of the SHA256 hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        raise IOError(f"Error reading file {filepath}: {e}") from e


def calculate_md5(filepath: Path) -> str:
    """
    Calculates the MD5 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        The hex digest of the MD5 hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    md5_hash = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)
        return md5_hash.hexdigest()
    except IOError as e:
        raise IOError(f"Error reading file {filepath}: {e}") from e


def calculate_sha1(filepath: Path) -> str:
    """
    Calculates the SHA1 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        The hex digest of the SHA1 hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    sha1_hash = hashlib.sha1()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha1_hash.update(byte_block)
        return sha1_hash.hexdigest()
    except IOError as e:
        raise IOError(f"Error reading file {filepath}: {e}") from e


def calculate_crc32(filepath: Path) -> int:
    """
    Calculates the CRC32 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        The CRC32 hash as an integer.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    import zlib

    crc32_hash = 0
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                crc32_hash = zlib.crc32(byte_block, crc32_hash)
        return crc32_hash
    except IOError as e:
        raise IOError(f"Error reading file {filepath}: {e}") from e


def calculate_hashes_from_stream(stream: BinaryIO, algorithms: List[str]) -> Dict[str, Any]:
    """
    Calculates multiple hashes from a stream in a single pass.

    Args:
        stream: A binary file-like object.
        algorithms: A list of hash algorithm names (e.g., ['md5', 'sha256', 'crc32']).

    Returns:
        A dictionary mapping algorithm names to their hash result (hex digest for most, int for crc32).
    """
    import zlib

    hashers = {alg: hashlib.new(alg) for alg in algorithms if alg != "crc32"}
    crc32_hash = 0 if "crc32" in algorithms else None

    stream.seek(0)
    while chunk := stream.read(4096):
        for hasher in hashers.values():
            hasher.update(chunk)
        if crc32_hash is not None:
            crc32_hash = zlib.crc32(chunk, crc32_hash)
    stream.seek(0)

    results = {name: hasher.hexdigest() for name, hasher in hashers.items()}
    if crc32_hash is not None:
        results["crc32"] = crc32_hash

    return results


# --- Async Wrappers ---


async def calculate_hashes_from_stream_async(stream: BinaryIO, algorithms: List[str]) -> Dict[str, str]:
    """Asynchronously calculates multiple hashes from a stream."""
    return await asyncio.to_thread(calculate_hashes_from_stream, stream, algorithms)


async def calculate_sha256_async(filepath: Path) -> str:
    """Asynchronously calculates the SHA256 hash of a file."""
    return await asyncio.to_thread(calculate_sha256, filepath)


async def calculate_md5_async(filepath: Path) -> str:
    """Asynchronously calculates the MD5 hash of a file."""
    return await asyncio.to_thread(calculate_md5, filepath)


async def calculate_sha1_async(filepath: Path) -> str:
    """Asynchronously calculates the SHA1 hash of a file."""
    return await asyncio.to_thread(calculate_sha1, filepath)


async def calculate_crc32_async(filepath: Path) -> int:
    """Asynchronously calculates the CRC32 hash of a file."""
    return await asyncio.to_thread(calculate_crc32, filepath)
