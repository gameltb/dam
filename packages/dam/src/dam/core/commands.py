from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, BinaryIO, Generic, Set, Tuple, TypeVar

from dam.core.enums import ExecutionStrategy
from dam.models.core.entity import Entity
from dam.system_events import BaseSystemEvent
from dam.utils.hash_utils import HashAlgorithm

if TYPE_CHECKING:
    from dam.core.world import World


ResultType = TypeVar("ResultType")
EventType = TypeVar("EventType", bound=BaseSystemEvent)


@dataclass
class BaseCommand(Generic[ResultType, EventType]):
    """Base class for all commands, which are requests that expect a response."""

    execution_strategy: ExecutionStrategy = field(kw_only=True, default=ExecutionStrategy.SERIAL)


@dataclass
class AddHashesFromStreamCommand(BaseCommand[None, BaseSystemEvent]):
    """Command to calculate and add multiple hash components to an entity from a stream."""

    entity_id: int
    stream: BinaryIO
    algorithms: Set[HashAlgorithm]


@dataclass
class GetOrCreateEntityFromStreamCommand(BaseCommand[Tuple[Entity, bytes], BaseSystemEvent]):
    """
    A command to get or create an entity from a stream.
    Returns a tuple of the entity and the calculated sha256 hash.
    """

    stream: BinaryIO


@dataclass
class EntityCommand(BaseCommand[ResultType, EventType]):
    """Base class for commands that operate on a single entity."""

    entity_id: int


@dataclass
class AnalysisCommand(EntityCommand[ResultType, EventType]):
    """Base class for commands that analyze an entity's data."""

    depth: int = 0
    stream: BinaryIO | None = None

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


__all__ = [
    "BaseCommand",
    "AddHashesFromStreamCommand",
    "GetOrCreateEntityFromStreamCommand",
    "EntityCommand",
    "AnalysisCommand",
]
