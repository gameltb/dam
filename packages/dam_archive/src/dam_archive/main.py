from typing import BinaryIO, Optional, Union

from .base import ArchiveHandler
from .registry import MIME_TYPE_HANDLERS


def open_archive(
    file_obj: Union[str, BinaryIO], mime_type: str, password: Optional[str] = None
) -> Optional[ArchiveHandler]:
    """
    Open an archive file with the appropriate handler.

    Args:
        file_obj: The file-like object or path to the archive.
        mime_type: The mime type of the archive, used to determine the handler.
        password: The password for the archive, if any.

    Returns:
        An instance of an ArchiveHandler, or None if no suitable handler is found.
    """
    handler_class = MIME_TYPE_HANDLERS.get(mime_type)
    if handler_class:
        return handler_class(file_obj, password)
    return None
