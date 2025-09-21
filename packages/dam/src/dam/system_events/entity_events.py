from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, BinaryIO, Optional

from dam.core.types import StreamProvider

from .base import BaseSystemEvent


@dataclass
class NewEntityCreatedEvent(BaseSystemEvent):
    """
    An event that is triggered when a new entity is created by a command.
    This is useful for recursive processing.
    """

    entity_id: int
    stream_provider: Optional[StreamProvider] = None
    filename: Optional[str] = None

    @asynccontextmanager
    async def open_stream(self) -> AsyncIterator[Optional[BinaryIO]]:
        """
        An async context manager that provides a fresh, readable binary stream if a provider exists.
        Usage:
            async with event.open_stream() as stream:
                if stream:
                    # do work
        """
        if not self.stream_provider:
            yield None
            return

        stream = self.stream_provider()
        try:
            yield stream
        finally:
            stream.close()
