import logging
from typing import IO, BinaryIO, Optional, Union

from .base import ArchiveHandler
from .exceptions import ArchiveError, PasswordRequiredError
from .registry import MIME_TYPE_HANDLERS

logger = logging.getLogger(__name__)


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
    handler_classes = MIME_TYPE_HANDLERS.get(mime_type)
    if not handler_classes:
        return None

    for handler_class in handler_classes:
        try:
            return handler_class(file_obj, password=password)
        except PasswordRequiredError:
            # Password errors should not be caught and suppressed, as they
            # indicate a fundamental issue that cannot be resolved by trying
            # a different handler.
            raise
        except ArchiveError as e:
            logger.warning(f"Handler {handler_class.__name__} failed: {e}")
            if isinstance(file_obj, IO):
                file_obj.seek(0)
        except Exception:
            logger.exception(f"An unexpected error occurred with handler {handler_class.__name__}")
            if isinstance(file_obj, IO):
                file_obj.seek(0)

    return None
