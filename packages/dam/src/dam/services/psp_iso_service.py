import hashlib
import zlib
from io import BytesIO
import pycdlib
import struct
from enum import IntEnum
from typing import Dict, Any, BinaryIO


class SFODataFormat:
    UTF8_NOTERM = 0x0004
    UTF8 = 0x0204
    INT32 = 0x0404


class SFOIndexTableEntry:
    def __init__(self, raw, offset):
        fields = struct.unpack('<HHIII', raw[offset: offset + 0x10])
        self.key_offset = fields[0]
        self.data_fmt = fields[1]
        self.data_len = fields[2]
        self.data_max_len = fields[3]
        self.data_offset = fields[4]


class SFO:
    def __init__(self, raw_sfo: bytes):
        if raw_sfo[:0x4] != b"\x00PSF":
            raise ValueError("Invalid SFO file format")

        version_minor = struct.unpack("<I", raw_sfo[0x5:0x8] + b"\x00")[0]
        self.version = f"{raw_sfo[0x04]}.{version_minor}"
        self.key_table_start, self.data_table_start, self.table_entries = struct.unpack('<III', raw_sfo[0x08:0x14])

        self.idx_table = [
            SFOIndexTableEntry(raw_sfo, 0x14 + idx * 0x10)
            for idx in range(self.table_entries)
        ]

        self.data: Dict[str, Any] = {}
        for i in range(len(self.idx_table)):
            self._read_entry(raw_sfo, i)

    def _read_entry(self, raw_sfo: bytes, idx: int):
        entry = self.idx_table[idx]

        k_start = self.key_table_start + entry.key_offset
        if idx == len(self.idx_table) - 1:
            k_end = self.data_table_start
        else:
            k_end = self.key_table_start + self.idx_table[idx + 1].key_offset
        key = raw_sfo[k_start:k_end].decode('utf-8', errors='ignore').rstrip("\x00")

        d_start = self.data_table_start + entry.data_offset
        d_end = d_start + entry.data_len

        if entry.data_fmt == SFODataFormat.INT32:
            data = int.from_bytes(raw_sfo[d_start:d_end], "little")
        else:
            data = raw_sfo[d_start:d_end].decode('utf-8', errors='ignore').rstrip("\x00")

        self.data[key] = data


def _calculate_hashes(stream: BinaryIO) -> Dict[str, bytes]:
    """Calculates MD5, SHA1, SHA256, and CRC32 hashes for a stream."""
    stream.seek(0)
    md5 = hashlib.md5()
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    crc32 = 0

    while chunk := stream.read(8192):
        md5.update(chunk)
        sha1.update(chunk)
        sha256.update(chunk)
        crc32 = zlib.crc32(chunk, crc32)

    stream.seek(0)

    return {
        "md5": md5.digest(),
        "sha1": sha1.digest(),
        "sha256": sha256.digest(),
        "crc32": crc32.to_bytes(4, 'big'),
    }


def process_iso_stream(stream: BinaryIO) -> Dict[str, Any]:
    """
    Processes a PSP ISO stream to extract SFO metadata and calculate hashes.

    Args:
        stream: A binary stream of the ISO file.

    Returns:
        A dictionary containing the hashes and SFO metadata.
    """
    hashes = _calculate_hashes(stream)

    iso = pycdlib.PyCdlib()
    try:
        iso.open_fp(stream)
    except Exception as e:
        # Could be a pycdlib error, e.g., not an ISO file
        raise IOError("Failed to open stream as ISO file") from e

    sfo_data = None
    try:
        for dirname, _, filelist in iso.walk(iso_path='/'):
            for file in filelist:
                if file.upper().endswith('PARAM.SFO'):
                    sfo_path = f"{dirname}/{file}".lstrip('/')
                    extracted_sfo = BytesIO()
                    iso.get_file_from_iso_fp(extracted_sfo, iso_path=sfo_path)
                    raw_sfo = extracted_sfo.getvalue()
                    try:
                        sfo = SFO(raw_sfo)
                        sfo_data = sfo.data
                    except Exception:
                        # Ignore SFO parsing errors
                        pass
                    break
            if sfo_data:
                break
    finally:
        iso.close()

    return {
        "hashes": hashes,
        "sfo_metadata": sfo_data,
    }
