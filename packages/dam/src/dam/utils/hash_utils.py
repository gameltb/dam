"""Utility functions for calculating content hashes."""

import hashlib
import zlib
from enum import Enum
from typing import Any, BinaryIO, Protocol, runtime_checkable

import blake3


@runtime_checkable
class Hasher(Protocol):
    """A protocol for hash-like objects."""

    def update(self, __data: bytes) -> Any:
        """Update the hash object with the bytes-like object."""
        ...

    def digest(self) -> bytes:
        """Return the digest of the data passed to the update() method so far."""
        ...


class HashAlgorithm(Enum):
    """An enumeration of supported hash algorithms."""

    SHA256 = "sha256"
    MD5 = "md5"
    SHA1 = "sha1"
    CRC32 = "crc32"
    BLAKE3 = "blake3"


def calculate_hashes_from_stream(stream: BinaryIO, algorithms: set[HashAlgorithm]) -> dict[HashAlgorithm, bytes | int]:
    """
    Calculate multiple hashes from a stream in a single pass.

    Args:
        stream: A binary file-like object.
        algorithms: A set of HashAlgorithm enums.

    Returns:
        A dictionary mapping algorithm enums to their hash result
        (bytes for most, int for crc32).

    """
    hashers: dict[HashAlgorithm, Hasher] = {
        alg: hashlib.new(alg.value) for alg in algorithms if alg not in [HashAlgorithm.CRC32, HashAlgorithm.BLAKE3]
    }
    if HashAlgorithm.BLAKE3 in algorithms:
        hashers[HashAlgorithm.BLAKE3] = blake3.blake3()

    crc32_hash: int | None = 0 if HashAlgorithm.CRC32 in algorithms else None

    # Inspired by hashlib.file_digest
    # Using read() for broader stream compatibility instead of readinto()
    chunk_size = 2**18
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break  # EOF
        for hasher in hashers.values():
            hasher.update(chunk)
        if crc32_hash is not None:
            crc32_hash = zlib.crc32(chunk, crc32_hash)

    results: dict[HashAlgorithm, bytes | int] = {name: hasher.digest() for name, hasher in hashers.items()}
    if crc32_hash is not None:
        results[HashAlgorithm.CRC32] = crc32_hash

    return results
