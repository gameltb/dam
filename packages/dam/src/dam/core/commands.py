from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, BinaryIO, Generic, Set, Tuple, TypeVar

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, BinaryIO, Generic, Set, Tuple, TypeVar

from dam.core.enums import ExecutionStrategy
from dam.models.core.entity import Entity
from dam.system_events.progress import (
    ProgressCompleted,
    ProgressError,
    ProgressStarted,
    ProgressUpdate,
)
from dam.utils.hash_utils import HashAlgorithm

if TYPE_CHECKING:
    from dam.core.world import World


ResultType = TypeVar("ResultType")


@dataclass
class BaseCommand(Generic[ResultType]):
    """Base class for all commands, which are requests that expect a response."""

    execution_strategy: ExecutionStrategy = field(kw_only=True, default=ExecutionStrategy.SERIAL)


@dataclass
class AddHashesFromStreamCommand(BaseCommand[None]):
    """Command to calculate and add multiple hash components to an entity from a stream."""

    entity_id: int
    stream: BinaryIO
    algorithms: Set[HashAlgorithm]


@dataclass
class GetOrCreateEntityFromStreamCommand(BaseCommand[Tuple[Entity, bytes]]):
    """
    A command to get or create an entity from a stream.
    Returns a tuple of the entity and the calculated sha256 hash.
    """

    stream: BinaryIO


@dataclass
class EntityCommand(BaseCommand[ResultType]):
    """Base class for commands that operate on a single entity."""

    entity_id: int


class _ProgressReporter:
    def __init__(self, world: "World"):
        self._world = world

    async def update(self, current: int | None = None, total: int | None = None, message: str | None = None):
        await self._world.dispatch_event(ProgressUpdate(current=current, total=total, message=message))


@dataclass
class AnalysisCommand(EntityCommand[ResultType]):
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

        stream = await world.dispatch_command(GetAssetStreamCommand(entity_id=self.entity_id)).get_first_non_none_value()
        if not stream:
            raise ValueError(f"Could not get asset stream for entity {self.entity_id}")
        return stream

    @asynccontextmanager
    async def progress_reporter(self, world: "World") -> AsyncGenerator[_ProgressReporter, None]:
        """A context manager to automatically handle progress reporting."""
        reporter = _ProgressReporter(world)
        try:
            await world.dispatch_event(ProgressStarted())
            yield reporter
        except Exception as e:
            await world.dispatch_event(ProgressError(exception=e))
            raise
        else:
            await world.dispatch_event(ProgressCompleted())


__all__ = [
    "BaseCommand",
    "AddHashesFromStreamCommand",
    "GetOrCreateEntityFromStreamCommand",
    "EntityCommand",
    "AnalysisCommand",
]
