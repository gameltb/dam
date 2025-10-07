import contextlib
import io
from collections.abc import Iterable, Sequence
from typing import IO, Any


class ChainedStream(io.IOBase):
    """
    A stream that concatenates multiple other binary streams, behaving like a single stream.
    This class is read-only.
    """

    def __init__(self, streams: Sequence[IO[bytes]]):
        super().__init__()
        self.streams = list(streams)
        self.stream_index = 0
        self._pos = 0

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        if self.stream_index >= len(self.streams):
            return b""

        if size == -1:
            data = b""
            while self.stream_index < len(self.streams):
                data += self.streams[self.stream_index].read()
                self.stream_index += 1
            self._pos += len(data)
            return data

        buffer = b""
        while len(buffer) < size and self.stream_index < len(self.streams):
            remaining = size - len(buffer)
            chunk = self.streams[self.stream_index].read(remaining)
            if not chunk:
                self.stream_index += 1
            buffer += chunk
        self._pos += len(buffer)
        return buffer

    def readline(self, size: int | None = -1) -> bytes:
        line = bytearray()
        limit = size if size is not None and size >= 0 else float("inf")
        while len(line) < limit:
            b = self.read(1)
            if not b:
                break
            line.extend(b)
            if b == b"\n":
                break
        return bytes(line)

    def readlines(self, hint: int = -1) -> list[bytes]:
        lines: list[bytes] = []
        while True:
            line = self.readline()
            if not line:
                break
            lines.append(line)
            if hint > 0 and sum(map(len, lines)) >= hint:
                break
        return lines

    def close(self) -> None:
        for stream in self.streams:
            with contextlib.suppress(Exception):
                stream.close()
        super().close()

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        raise io.UnsupportedOperation("seek is not supported")

    def write(self, s: bytes) -> int:
        raise io.UnsupportedOperation("write is not supported")

    def writelines(self, lines: Iterable[Any]) -> None:
        raise io.UnsupportedOperation("writelines is not supported")

    def truncate(self, size: int | None = None) -> int:
        raise io.UnsupportedOperation("truncate is not supported")

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("ChainedStream has no file descriptor")
