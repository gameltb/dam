import hashlib
import zlib
from io import BytesIO

import blake3

from dam.utils.hash_utils import HashAlgorithm, calculate_hashes_from_stream


def test_calculate_hashes_from_stream() -> None:
    """
    Tests that calculate_hashes_from_stream correctly calculates multiple hashes.
    """
    data = b"hello world"
    stream = BytesIO(data)
    algorithms = {
        HashAlgorithm.SHA256,
        HashAlgorithm.MD5,
        HashAlgorithm.BLAKE3,
        HashAlgorithm.CRC32,
    }

    hashes = calculate_hashes_from_stream(stream, algorithms)

    assert hashes[HashAlgorithm.SHA256] == hashlib.sha256(data).digest()
    assert hashes[HashAlgorithm.MD5] == hashlib.md5(data).digest()
    assert hashes[HashAlgorithm.BLAKE3] == blake3.blake3(data).digest()
    assert hashes[HashAlgorithm.CRC32] == zlib.crc32(data)
