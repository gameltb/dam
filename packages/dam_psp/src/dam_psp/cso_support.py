"""Provides a file-like object for reading CSO (Compressed ISO) files."""
import io
import struct
import zlib
from typing import BinaryIO

CISO_MAGIC = 0x4F534943  # "CISO"
CISO_HEADER_FMT = "<LLQLBBxx"
CISO_HEADER_SIZE = 24
CISO_WBITS = -15  # Raw deflate stream


class CsoDecompressor(io.RawIOBase):
    """A file-like object for reading decompressed data from a CSO (Compressed ISO) stream."""

    def __init__(self, cso_stream: BinaryIO):
        """Initialize the CSO decompressor, read and parse the header."""
        super().__init__()
        self._cso_stream = cso_stream
        self._cso_stream.seek(0)

        header_data = self._cso_stream.read(CISO_HEADER_SIZE)
        if len(header_data) != CISO_HEADER_SIZE:
            raise OSError("Invalid CSO header: not enough data")

        (
            magic,
            _,  # header_size
            self._uncompressed_size,
            self._block_size,
            ver,
            self._align,
        ) = struct.unpack(CISO_HEADER_FMT, header_data)

        if magic != CISO_MAGIC:
            raise OSError("Invalid CSO magic number")
        if ver > 1:
            raise NotImplementedError(f"CSO version {ver} is not supported")

        self._total_blocks = self._uncompressed_size // self._block_size
        index_size = (self._total_blocks + 1) * 4

        index_data = self._cso_stream.read(index_size)
        if len(index_data) != index_size:
            raise OSError("Invalid CSO block index: not enough data")
        self._block_index = struct.unpack(f"<{self._total_blocks + 1}I", index_data)

        self._position = 0
        self._current_block_num = -1
        self._current_block_data = b""
        self._current_block_pos = 0

    def readable(self) -> bool:
        """Return True if the stream can be read from."""
        return True

    def seekable(self) -> bool:
        """Return True if the stream supports random access."""
        return True

    def tell(self) -> int:
        """Return the current stream position."""
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Change the stream position to the given byte offset."""
        if whence == io.SEEK_SET:
            self._position = offset
        elif whence == io.SEEK_CUR:
            self._position += offset
        elif whence == io.SEEK_END:
            self._position = self._uncompressed_size + offset
        else:
            raise ValueError(f"Invalid whence value: {whence}")

        self._current_block_num = -1
        self._current_block_data = b""
        self._current_block_pos = 0

        return self._position

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes from the object and return them."""
        if size == -1:
            size = self._uncompressed_size - self._position

        if self._position >= self._uncompressed_size:
            return b""

        remaining_size = size
        result = bytearray()

        while remaining_size > 0 and self._position < self._uncompressed_size:
            block_num = self._position // self._block_size
            offset_in_block = self._position % self._block_size

            if block_num != self._current_block_num:
                self._load_block(block_num)
                self._current_block_pos = 0

            if self._current_block_pos != offset_in_block:
                self._current_block_pos = offset_in_block

            chunk_to_read = min(remaining_size, self._block_size - self._current_block_pos)
            result.extend(self._current_block_data[self._current_block_pos : self._current_block_pos + chunk_to_read])

            bytes_read = len(result) - (size - remaining_size)
            self._current_block_pos += bytes_read
            self._position += bytes_read
            remaining_size -= bytes_read

        return bytes(result)

    def _load_block(self, block_num: int):
        """Load and decompress a block of data from the CSO stream."""
        if block_num >= self._total_blocks:
            raise OSError("Attempt to read past the end of the file")

        index_entry = self._block_index[block_num]
        is_plain = index_entry & 0x80000000
        offset = (index_entry & 0x7FFFFFFF) << self._align

        self._cso_stream.seek(offset)

        if is_plain:
            self._current_block_data = self._cso_stream.read(self._block_size)
        else:
            next_index_entry = self._block_index[block_num + 1]
            next_offset = (next_index_entry & 0x7FFFFFFF) << self._align
            read_size = next_offset - offset
            compressed_data = self._cso_stream.read(read_size)
            self._current_block_data = zlib.decompress(compressed_data, wbits=CISO_WBITS)

        if len(self._current_block_data) < self._block_size and block_num < self._total_blocks - 1:
            padding = self._block_size - len(self._current_block_data)
            self._current_block_data += b"\x00" * padding

        self._current_block_num = block_num
