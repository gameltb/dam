from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, BinaryIO, Dict, List, Optional

from dam.commands.core import EntityCommand, EventType, ResultType
from dam.core.types import StreamProvider
from dam.system_events.base import BaseSystemEvent

if TYPE_CHECKING:
    from dam.core.world import World


@dataclass
class AnalysisCommand(EntityCommand[ResultType, EventType]):
    """Base class for commands that analyze an entity's data."""

    stream_provider: Optional[StreamProvider] = None

    @classmethod
    def get_supported_types(cls) -> Dict[str, List[str]]:
        """
        Returns a dictionary of supported MIME types and file extensions.
        Example: {"mimetypes": ["image/jpeg"], "extensions": [".jpg", ".jpeg"]}
        """
        return {"mimetypes": [], "extensions": []}

    async def get_stream_provider(self, world: "World") -> Optional[StreamProvider]:
        """
        Gets a provider for a binary stream for the command's entity.
        If a provider was passed in the command, it is returned.
        Otherwise, a new GetAssetStreamCommand is dispatched to fetch the provider.
        """
        if self.stream_provider:
            return self.stream_provider

        from dam.commands.asset_commands import GetAssetStreamCommand

        provider = await world.dispatch_command(
            GetAssetStreamCommand(entity_id=self.entity_id)
        ).get_first_non_none_value()
        return provider

    @asynccontextmanager
    async def open_stream(self, world: "World") -> AsyncIterator[Optional[BinaryIO]]:
        """
        An async context manager that provides a fresh, readable binary stream for the entity.
        It fetches a stream provider if one is not already available.
        Usage:
            async with cmd.open_stream(world) as stream:
                if stream:
                    # do work
        """
        provider = await self.get_stream_provider(world)

        if not provider:
            yield None
            return

        stream = provider()
        try:
            yield stream
        finally:
            stream.close()


@dataclass
class AutoSetMimeTypeCommand(AnalysisCommand[None, BaseSystemEvent]):
    """
    A command to automatically set the mime type for an asset.
    """

    entity_id: int
