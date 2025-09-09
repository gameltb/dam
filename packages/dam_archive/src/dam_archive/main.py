from typing import Any, List, Optional

from . import rar, sevenzip, zip
from .base import ArchiveHandler
from .registry import get_handlers


def register_all_handlers() -> None:
    """
    Registers all available archive handlers.
    """
    zip.register()
    sevenzip.register()
    rar.register()


def open_archive(file_obj: Any, filename: str, passwords: Optional[List[str]] = None) -> Optional[ArchiveHandler]:
    """
    Open an archive file with the appropriate handler.

    Args:
        file_obj: The file-like object or path to the archive.
        filename: The filename of the archive, used to determine the handler.
        passwords: The passwords for the archive, if any.

    Returns:
        An instance of an ArchiveHandler, or None if no suitable handler is found.
    """
    for handler_class in get_handlers():
        if handler_class.can_handle(filename):
            return handler_class(file_obj, passwords)
    return None


register_all_handlers()
