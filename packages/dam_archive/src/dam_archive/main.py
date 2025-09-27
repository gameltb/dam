import logging
from typing import BinaryIO, Optional, Union

from .base import ArchiveHandler, StreamProvider, to_stream_provider
from .exceptions import ArchiveError, PasswordRequiredError
from .registry import MIME_TYPE_HANDLERS

logger = logging.getLogger(__name__)


def open_archive(
    file_or_path_or_provider: Union[str, BinaryIO, StreamProvider], mime_type: str, password: Optional[str] = None
) -> Optional[ArchiveHandler]:
    """
    Open an archive file with the appropriate handler.

    Args:
        file_or_path_or_provider: The file-like object, path to the archive, or a stream provider.
        mime_type: The mime type of the archive, used to determine the handler.
        password: The password for the archive, if any.

    Returns:
        An instance of an ArchiveHandler, or None if no suitable handler is found.
    """
    handler_classes = MIME_TYPE_HANDLERS.get(mime_type)
    if not handler_classes:
        return None

    stream_provider: StreamProvider
    if callable(file_or_path_or_provider):
        stream_provider = file_or_path_or_provider
    else:
        try:
            stream_provider = to_stream_provider(file_or_path_or_provider)
        except ValueError:
            return None

    for handler_class in handler_classes:
        try:
            return handler_class(stream_provider, password=password)
        except PasswordRequiredError:
            # Password errors should not be caught and suppressed, as they
            # indicate a fundamental issue that cannot be resolved by trying
            # a different handler.
            raise
        except ArchiveError as e:
            logger.warning(f"Handler {handler_class.__name__} failed: {e}")
        except Exception:
            logger.exception(f"An unexpected error occurred with handler {handler_class.__name__}")

    return None
