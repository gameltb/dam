import struct
from typing import Any, BinaryIO

class GGUFMetadataReader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.fields: dict[str, Any] = {}
        self.tensor_count = 0
        self._read_metadata()

    def _read_struct(self, f: BinaryIO, fmt: str) -> tuple:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, f.read(size))

    def _read_string(self, f: BinaryIO) -> str:
        (length,) = self._read_struct(f, "<Q")
        return f.read(length).decode("utf-8")

    def _read_value(self, f: BinaryIO, value_type: int) -> Any:
        if value_type == 0:  # UINT8
            return self._read_struct(f, "<B")[0]
        if value_type == 1:  # INT8
            return self._read_struct(f, "<b")[0]
        if value_type == 2:  # UINT16
            return self._read_struct(f, "<H")[0]
        if value_type == 3:  # INT16
            return self._read_struct(f, "<h")[0]
        if value_type == 4:  # UINT32
            return self._read_struct(f, "<I")[0]
        if value_type == 5:  # INT32
            return self._read_struct(f, "<i")[0]
        if value_type == 6:  # FLOAT32
            return self._read_struct(f, "<f")[0]
        if value_type == 7:  # BOOL
            return self._read_struct(f, "<?")[0]
        if value_type == 8:  # STRING
            return self._read_string(f)
        if value_type == 9:  # ARRAY
            (array_type, count) = self._read_struct(f, "<IQ")
            return [self._read_value(f, array_type) for _ in range(count)]
        if value_type == 10:  # UINT64
            return self._read_struct(f, "<Q")[0]
        if value_type == 11:  # INT64
            return self._read_struct(f, "<q")[0]
        if value_type == 12:  # FLOAT64
            return self._read_struct(f, "<d")[0]
        raise Exception(f"Unknown value type: {value_type}")

    def _read_metadata(self):
        with open(self.filepath, "rb") as f:
            (magic,) = self._read_struct(f, "<I")
            if magic != 0x46554747:  # GGUF
                raise Exception("Not a GGUF file")

            (version, self.tensor_count, kv_count) = self._read_struct(f, "<IQQ")

            for _ in range(kv_count):
                key = self._read_string(f)
                (value_type,) = self._read_struct(f, "<I")
                value = self._read_value(f, value_type)
                self.fields[key] = value

    def get_field(self, key: str) -> Any | None:
        return self.fields.get(key)

    def get_tensor_count(self) -> int:
        return self.tensor_count
