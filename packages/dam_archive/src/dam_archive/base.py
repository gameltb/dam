from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import BinaryIO, Callable, Iterator, List, Optional, Tuple, Union

StreamProvider = Callable[[], BinaryIO]


def to_stream_provider(
    file_or_path: Union[str, BinaryIO],
) -> StreamProvider:
    if isinstance(file_or_path, str):
        return lambda: open(file_or_path, "rb")
    else:
        # If the stream is seekable, we can reuse it.
        if file_or_path.seekable():

            def seekable_stream_provider() -> BinaryIO:
                file_or_path.seek(0)
                return file_or_path

            return seekable_stream_provider
        else:
            # If the stream is not seekable, we cannot create a reliable
            # stream provider from it.
            raise ValueError("Cannot create a stream provider from a non-seekable stream.")


@dataclass
class ArchiveMemberInfo:
    """
    Represents information about a member in an archive.
    """

    name: str
    size: int
    modified_at: Optional[datetime]


class ArchiveHandler(ABC):
    """
    Abstract base class for archive handlers.
    """

    def __init__(self, stream_provider: StreamProvider, password: Optional[str] = None):
        """
        Initializes the archive handler.

        Args:
            stream_provider: A function that returns a new stream to the archive.
            password: The password for the archive, if any.
        """
        self._stream_provider = stream_provider
        self.password = password

    @property
    def comment(self) -> Optional[str]:
        """The comment of the archive, if any."""
        return None

    @abstractmethod
    def list_files(self) -> List[ArchiveMemberInfo]:
        """List all file names and sizes in the archive."""
        pass

    @abstractmethod
    def iter_files(self) -> Iterator[Tuple[ArchiveMemberInfo, BinaryIO]]:
        """
        Iterate over all files in the archive in their natural order.

        This method is designed for efficient, sequential processing of archive
        members. It yields tuples of (`ArchiveMemberInfo`, `BinaryIO`), which can be
        used to access member information and open a stream to the file's content.

        For archive formats that support it, this method should be implemented
        to stream data from the archive rather than performing random-access reads,
        which can be inefficient, especially for solid archives.
        """
        pass

    @abstractmethod
    def open_file(self, file_name: str) -> Tuple[ArchiveMemberInfo, BinaryIO]:
        """Open a specific file from the archive and return a file-like object."""

        pass

    @abstractmethod
    def close(self) -> None:
        """
        Closes the archive file and releases any resources.
        This should be called when the handler is no longer needed.
        """
        pass
