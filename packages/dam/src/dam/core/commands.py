from dataclasses import dataclass, field
from typing import IO, Generic, List, Set, TypeVar

from dam.utils.hash_utils import HashAlgorithm

ResultType = TypeVar("ResultType")


@dataclass
class CommandResult(Generic[ResultType]):
    """A container for the results of a command, collecting results from all handlers."""

    results: List[ResultType] = field(default_factory=list)


@dataclass
class BaseCommand(Generic[ResultType]):
    """Base class for all commands, which are requests that expect a response."""

    pass


@dataclass
class AddHashesFromStreamCommand(BaseCommand[None]):
    """Command to calculate and add multiple hash components to an entity from a stream."""

    entity_id: int
    stream: IO[bytes]
    algorithms: Set[HashAlgorithm]


from typing import Tuple

from dam.models.core.entity import Entity


@dataclass
class GetOrCreateEntityFromStreamCommand(BaseCommand[Tuple[Entity, bytes]]):
    """
    A command to get or create an entity from a stream.
    Returns a tuple of the entity and the calculated sha256 hash.
    """

    stream: IO[bytes]


__all__ = ["BaseCommand", "CommandResult", "AddHashesFromStreamCommand", "GetOrCreateEntityFromStreamCommand"]
