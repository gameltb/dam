from typing import override, Union, Optional
from py7zr import Py7zIO, WriterFactory
import io

class InMemoryIO(Py7zIO):
    def __init__(self):
        self._buffer = io.BytesIO()

    @override
    def write(self, s: Union[bytes, bytearray]):
        self._buffer.write(s)

    @override
    def read(self, size: Optional[int] = None) -> bytes:
        return self._buffer.read(size)

    @override
    def seek(self, offset: int, whence: int = 0) -> int:
        return self._buffer.seek(offset, whence)

    @override
    def flush(self) -> None:
        self._buffer.flush()

    @property
    @override
    def size(self) -> int:
        return len(self.getbuffer())

    def getbuffer(self) -> memoryview:
        return self._buffer.getbuffer()

    def get_bytes_io(self) -> io.BytesIO:
        self._buffer.seek(0)
        return self._buffer


class InMemoryIOFactory(WriterFactory):
    def __init__(self):
        self.products = {}

    @override
    def create(self, filename: str) -> Py7zIO:
        product = InMemoryIO()
        self.products[filename] = product
        return product
