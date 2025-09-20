from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO, Dict, List

from dam.commands.core import EntityCommand, EventType, ResultType
from dam.system_events import BaseSystemEvent

if TYPE_CHECKING:
    from dam.core.world import World


@dataclass
class AnalysisCommand(EntityCommand[ResultType, EventType]):
    """Base class for commands that analyze an entity's data."""

    stream: BinaryIO | None = None

    @classmethod
    def get_supported_types(cls) -> Dict[str, List[str]]:
        """
        Returns a dictionary of supported MIME types and file extensions.
        Example: {"mimetypes": ["image/jpeg"], "extensions": [".jpg", ".jpeg"]}
        """
        return {"mimetypes": [], "extensions": []}

    async def get_stream(self, world: "World") -> BinaryIO:
        """
        Gets a readable, seekable binary stream for the command's entity.
        If a stream was provided in the command, it is returned.
        Otherwise, a new GetAssetStreamCommand is dispatched to fetch the stream.
        """
        if self.stream:
            return self.stream

        from dam.commands.asset_commands import GetAssetStreamCommand

        stream = await world.dispatch_command(
            GetAssetStreamCommand(entity_id=self.entity_id)
        ).get_first_non_none_value()
        if not stream:
            raise ValueError(f"Could not get asset stream for entity {self.entity_id}")
        return stream


@dataclass
class AutoSetMimeTypeCommand(AnalysisCommand[None, BaseSystemEvent]):
    """
    A command to automatically set the mime type for an asset.
    """

    entity_id: int
