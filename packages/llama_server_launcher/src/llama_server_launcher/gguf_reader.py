import struct
from pathlib import Path
from typing import Any, BinaryIO


class GGUFMetadataReader:
    """Reads GGUF file metadata."""

    def __init__(self, filepath: str):
        """Initialise the GGUF metadata reader."""
        self.filepath = filepath
        self.fields: dict[str, Any] = {}
        self.tensor_count = 0
        self._read_metadata()

    def _read_struct(self, f: BinaryIO, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, f.read(size))

    def _read_string(self, f: BinaryIO) -> str:
        (length,) = self._read_struct(f, "<Q")
        return f.read(length).decode("utf-8")

    def _read_value(self, f: BinaryIO, value_type: int) -> Any:
        """Read a value of a given type from the file."""
        type_map: dict[int, Any] = {
            0: lambda: self._read_struct(f, "<B")[0],  # UINT8
            1: lambda: self._read_struct(f, "<b")[0],  # INT8
            2: lambda: self._read_struct(f, "<H")[0],  # UINT16
            3: lambda: self._read_struct(f, "<h")[0],  # INT16
            4: lambda: self._read_struct(f, "<I")[0],  # UINT32
            5: lambda: self._read_struct(f, "<i")[0],  # INT32
            6: lambda: self._read_struct(f, "<f")[0],  # FLOAT32
            7: lambda: self._read_struct(f, "<?")[0],  # BOOL
            8: lambda: self._read_string(f),  # STRING
            9: lambda: [
                self._read_value(f, self._read_struct(f, "<I")[0]) for _ in range(self._read_struct(f, "<Q")[0])
            ],  # ARRAY
            10: lambda: self._read_struct(f, "<Q")[0],  # UINT64
            11: lambda: self._read_struct(f, "<q")[0],  # INT64
            12: lambda: self._read_struct(f, "<d")[0],  # FLOAT64
        }
        if value_type in type_map:
            return type_map[value_type]()
        raise Exception(f"Unknown value type: {value_type}")

    def _read_metadata(self):
        """Read the GGUF file metadata."""
        gguf_magic = 0x46554747
        with Path(self.filepath).open("rb") as f:
            (magic,) = self._read_struct(f, "<I")
            if magic != gguf_magic:
                raise Exception("Not a GGUF file")

            (_version, self.tensor_count, kv_count) = self._read_struct(f, "<IQQ")

            for _ in range(kv_count):
                key = self._read_string(f)
                (value_type,) = self._read_struct(f, "<I")
                value = self._read_value(f, value_type)
                self.fields[key] = value

    def get_field(self, key: str) -> Any | None:
        """Return the value of a metadata field."""
        return self.fields.get(key)

    def get_tensor_count(self) -> int:
        """Return the number of tensors."""
        return self.tensor_count
