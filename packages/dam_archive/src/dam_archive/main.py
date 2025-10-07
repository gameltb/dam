import logging
from pathlib import Path

from dam.core.types import StreamProvider

from .base import ArchiveHandler, to_stream_provider
from .exceptions import ArchiveError, PasswordRequiredError
from .registry import MIME_TYPE_HANDLERS

logger = logging.getLogger(__name__)


async def open_archive(
    file_or_path_or_provider: str | Path | StreamProvider,
    mime_type: str,
    password: str | None = None,
) -> ArchiveHandler | None:
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
    if isinstance(file_or_path_or_provider, StreamProvider):
        stream_provider = file_or_path_or_provider
    else:
        try:
            stream_provider = to_stream_provider(file_or_path_or_provider)
        except ValueError:
            return None

    for handler_class in handler_classes:
        try:
            return await handler_class.create(stream_provider, password=password)
        except PasswordRequiredError:
            # Password errors should not be caught and suppressed, as they
            # indicate a fundamental issue that cannot be resolved by trying
            # a different handler.
            raise
        except ArchiveError as e:
            logger.warning("Handler %s failed: %s", handler_class.__name__, e)
        except Exception:
            logger.exception("An unexpected error occurred with handler %s", handler_class.__name__)

    return None
