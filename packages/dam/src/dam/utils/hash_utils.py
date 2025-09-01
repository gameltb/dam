import hashlib
import zlib
from enum import Enum
from typing import IO, Dict, Set

import blake3


class HashAlgorithm(Enum):
    SHA256 = "sha256"
    MD5 = "md5"
    SHA1 = "sha1"
    CRC32 = "crc32"
    BLAKE3 = "blake3"


def calculate_hashes_from_stream(
    stream: IO[bytes], algorithms: Set[HashAlgorithm]
) -> Dict[HashAlgorithm, bytes | int]:
    """
    Calculates multiple hashes from a stream in a single pass.

    Args:
        stream: A binary file-like object.
        algorithms: A set of HashAlgorithm enums.

    Returns:
        A dictionary mapping algorithm enums to their hash result
        (bytes for most, int for crc32).
    """
    hashers = {
        alg: hashlib.new(alg.value)
        for alg in algorithms
        if alg not in [HashAlgorithm.CRC32, HashAlgorithm.BLAKE3]
    }
    if HashAlgorithm.BLAKE3 in algorithms:
        hashers[HashAlgorithm.BLAKE3] = blake3.blake3()

    crc32_hash = 0 if HashAlgorithm.CRC32 in algorithms else None

    # Inspired by hashlib.file_digest
    buf = bytearray(2**18)  # Reusable buffer to reduce allocations.
    view = memoryview(buf)
    while True:
        size = stream.readinto(buf)
        if size == 0:
            break  # EOF
        for hasher in hashers.values():
            hasher.update(view[:size])
        if crc32_hash is not None:
            crc32_hash = zlib.crc32(view[:size], crc32_hash)

    results = {name: hasher.digest() for name, hasher in hashers.items()}
    if crc32_hash is not None:
        results[HashAlgorithm.CRC32] = crc32_hash

    return results
