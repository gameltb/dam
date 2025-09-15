from dataclasses import dataclass, field
from typing import BinaryIO, Generic, Set, Tuple, TypeVar

from dam.core.enums import ExecutionStrategy
from dam.models.core.entity import Entity
from dam.utils.hash_utils import HashAlgorithm

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


__all__ = ["BaseCommand", "AddHashesFromStreamCommand", "GetOrCreateEntityFromStreamCommand"]
