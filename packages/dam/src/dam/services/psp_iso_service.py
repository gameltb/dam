from io import BytesIO
import pycdlib
import struct
from enum import IntEnum
from typing import Dict, Any, BinaryIO
from pathlib import Path
from dam.services import hashing_service


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
        self.raw_data: Dict[str, Any] = {}
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

        raw_data_bytes = raw_sfo[d_start:d_end]
        if entry.data_fmt == SFODataFormat.INT32:
            data = int.from_bytes(raw_data_bytes, "little")
            self.raw_data[key] = int.from_bytes(raw_data_bytes, "little")
        else:
            data = raw_data_bytes.decode('utf-8', errors='ignore').rstrip("\x00")
            self.raw_data[key] = raw_data_bytes.rstrip(b'\x00').decode('utf-8', errors='ignore')

        self.data[key] = data


def _calculate_hashes(stream: BinaryIO) -> Dict[str, bytes]:
    """Calculates MD5, SHA1, SHA256, and CRC32 hashes for a stream."""
    # This function now delegates to the hashing_service.
    # The hashing_service works with file paths, so we need to write the stream to a temporary file.
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        stream.seek(0)
        tmp.write(stream.read())
        tmp_path_str = tmp.name

    stream.seek(0)
    tmp_path = Path(tmp_path_str)

    try:
        md5_hex = hashing_service.calculate_md5(tmp_path)
        sha1_hex = hashing_service.calculate_sha1(tmp_path)
        sha256_hex = hashing_service.calculate_sha256(tmp_path)
        crc32_int = hashing_service.calculate_crc32(tmp_path)
    finally:
        os.unlink(tmp_path_str)

    return {
        "md5": bytes.fromhex(md5_hex),
        "sha1": bytes.fromhex(sha1_hex),
        "sha256": bytes.fromhex(sha256_hex),
        "crc32": crc32_int.to_bytes(4, 'big'),
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
    sfo_raw_data = None
    try:
        for dirname, _, filelist in iso.walk(iso_path='/'):
            for file in filelist:
                if file.upper().startswith('PARAM.SFO'):
                    sfo_path = f"/{dirname}/{file}".replace('//', '/')
                    extracted_sfo = BytesIO()
                    iso.get_file_from_iso_fp(extracted_sfo, iso_path=sfo_path)
                    raw_sfo = extracted_sfo.getvalue()
                    try:
                        sfo = SFO(raw_sfo)
                        sfo_data = sfo.data
                        sfo_raw_data = sfo.raw_data
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
        "sfo_raw_metadata": sfo_raw_data,
    }
