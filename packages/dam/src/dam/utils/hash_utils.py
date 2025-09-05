import hashlib
import zlib
from enum import Enum
from typing import IO, Any, Dict, Protocol, Set, runtime_checkable

import blake3


@runtime_checkable
class Hasher(Protocol):
    def update(self, __data: bytes) -> Any: ...
    def digest(self) -> bytes: ...


class HashAlgorithm(Enum):
    SHA256 = "sha256"
    MD5 = "md5"
    SHA1 = "sha1"
    CRC32 = "crc32"
    BLAKE3 = "blake3"


def calculate_hashes_from_stream(stream: IO[bytes], algorithms: Set[HashAlgorithm]) -> Dict[HashAlgorithm, bytes | int]:
    """
    Calculates multiple hashes from a stream in a single pass.

    Args:
        stream: A binary file-like object.
        algorithms: A set of HashAlgorithm enums.

    Returns:
        A dictionary mapping algorithm enums to their hash result
        (bytes for most, int for crc32).
    """
    hashers: Dict[HashAlgorithm, Hasher] = {
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

    results: Dict[HashAlgorithm, bytes | int] = {name: hasher.digest() for name, hasher in hashers.items()}
    if crc32_hash is not None:
        results[HashAlgorithm.CRC32] = crc32_hash

    return results
