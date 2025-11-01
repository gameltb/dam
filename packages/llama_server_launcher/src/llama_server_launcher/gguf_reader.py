"""A minimal GGUF metadata reader."""

import struct
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO

# GGUF magic number
GGUF_MAGIC = 0x47475546


class GGUFValueType(IntEnum):
    """GGUF value types."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGUFMetadataReader:
    """A minimal GGUF metadata reader."""

    def __init__(self, filepath: str) -> None:
        """Initialise the GGUF metadata reader."""
        self.filepath = Path(filepath)
        self.fields: dict[str, Any] = {}
        self.tensor_count = 0
        self._read_metadata()

    def _read_struct(self, f: BinaryIO, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, f.read(size))

    def _read_string(self, f: BinaryIO) -> str:
        (length,) = self._read_struct(f, "<Q")
        return f.read(int(length)).decode("utf-8")

    def _read_value(self, f: BinaryIO, value_type: int) -> Any:  # noqa: PLR0911, PLR0912
        if value_type == GGUFValueType.UINT8:
            return self._read_struct(f, "<B")[0]
        if value_type == GGUFValueType.INT8:
            return self._read_struct(f, "<b")[0]
        if value_type == GGUFValueType.UINT16:
            return self._read_struct(f, "<H")[0]
        if value_type == GGUFValueType.INT16:
            return self._read_struct(f, "<h")[0]
        if value_type == GGUFValueType.UINT32:
            return self._read_struct(f, "<I")[0]
        if value_type == GGUFValueType.INT32:
            return self._read_struct(f, "<i")[0]
        if value_type == GGUFValueType.FLOAT32:
            return self._read_struct(f, "<f")[0]
        if value_type == GGUFValueType.BOOL:
            return self._read_struct(f, "<?")[0]
        if value_type == GGUFValueType.STRING:
            return self._read_string(f)
        if value_type == GGUFValueType.ARRAY:
            (array_type, count) = self._read_struct(f, "<IQ")
            return [self._read_value(f, array_type) for _ in range(count)]
        if value_type == GGUFValueType.UINT64:
            return self._read_struct(f, "<Q")[0]
        if value_type == GGUFValueType.INT64:
            return self._read_struct(f, "<q")[0]
        if value_type == GGUFValueType.FLOAT64:
            return self._read_struct(f, "<d")[0]
        raise Exception(f"Unknown value type: {value_type}")

    def _read_metadata(self) -> None:
        with self.filepath.open("rb") as f:
            (magic,) = self._read_struct(f, "<I")
            if magic != GGUF_MAGIC:
                raise Exception("Not a GGUF file")

            (_version, self.tensor_count, kv_count) = self._read_struct(f, "<IQQ")

            for _ in range(int(kv_count)):
                key = self._read_string(f)
                (value_type,) = self._read_struct(f, "<I")
                value = self._read_value(f, value_type)
                self.fields[key] = value

    def get_field(self, key: str) -> Any | None:
        """Get a metadata field by key."""
        return self.fields.get(key)

    def get_tensor_count(self) -> int:
        """Get the number of tensors."""
        return self.tensor_count
