"""Commands for analyzing entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.types import StreamProvider
from ..system_events.base import BaseSystemEvent
from .asset_commands import GetAssetStreamCommand
from .core import EntityCommand, EventType, ResultType

if TYPE_CHECKING:
    from ..core.world import World


@dataclass
class AnalysisCommand(EntityCommand[ResultType, EventType]):
    """Base class for commands that analyze an entity's data."""

    stream_provider: StreamProvider | None = None

    @classmethod
    def get_supported_types(cls) -> dict[str, list[str]]:
        """
        Return a dictionary of supported MIME types and file extensions.

        Example: {"mimetypes": ["image/jpeg"], "extensions": [".jpg", ".jpeg"]}.
        """
        return {"mimetypes": [], "extensions": []}

    async def get_stream_provider(self, world: World) -> StreamProvider | None:
        """
        Get a provider for a binary stream for the command's entity.

        If a provider was passed in the command, it is returned.
        Otherwise, a new GetAssetStreamCommand is dispatched to fetch the provider.
        """
        if self.stream_provider:
            return self.stream_provider

        all_providers = await world.dispatch_command(GetAssetStreamCommand(entity_id=self.entity_id)).get_all_results()

        valid_providers = [p for p in all_providers if p is not None]
        if not valid_providers:
            return None

        return valid_providers[0]


@dataclass
class AutoSetMimeTypeCommand(AnalysisCommand[None, BaseSystemEvent]):
    """A command to automatically set the mime type for an asset."""

    entity_id: int
