from dataclasses import dataclass, field
from typing import Any, BinaryIO, Generic, Iterator, List, Set, Tuple, TypeVar

from dam.core.result import HandlerResult
from dam.models.core.entity import Entity
from dam.utils.hash_utils import HashAlgorithm

ResultType = TypeVar("ResultType")


@dataclass
class CommandResult(Generic[ResultType]):
    """
    A container for the results of a command, collecting results from all handlers.
    Each result is wrapped in a HandlerResult, which can be either Ok or Err.
    """

    results: List[HandlerResult[ResultType]] = field(default_factory=list)  # type: ignore

    def __iter__(self) -> Iterator[HandlerResult[ResultType]]:
        """Iterates over the HandlerResult objects."""
        return iter(self.results)

    def iter_ok_values(self) -> Iterator[ResultType]:
        """Iterates over the values of successful results, skipping any errors."""
        for res in self.results:
            if res.is_ok():
                yield res.unwrap()

    def get_first_ok_value(self) -> ResultType:
        """
        Returns the first successful result value.
        Raises ValueError if no successful results are found.
        """
        for value in self.iter_ok_values():
            return value
        raise ValueError("No successful results found.")

    def get_first_non_none_value(self) -> ResultType:
        """
        Returns the first successful result value that is not None.
        Raises ValueError if no such result is found.
        """
        for value in self.iter_ok_values():
            if value is not None:
                return value
        raise ValueError("No successful, non-None results found.")

    def get_one_value(self) -> ResultType:
        """
        Returns the single successful result value.
        Raises ValueError if there are zero or more than one successful results.
        """
        ok_values = list(self.iter_ok_values())
        if len(ok_values) == 0:
            raise ValueError("Expected one result, but found none.")
        if len(ok_values) > 1:
            raise ValueError(f"Expected one result, but found {len(ok_values)}.")
        return ok_values[0]

    def iter_ok_values_flat(self) -> Iterator[Any]:
        """
        Iterates over successful results and "flattens" them.
        If a successful result value is a list, it yields each item from the list.
        Otherwise, it yields the value itself.
        """
        for res in self.results:
            if res.is_ok():
                value = res.unwrap()
                if isinstance(value, list):
                    yield from value
                elif value is not None:
                    yield value


@dataclass
class BaseCommand(Generic[ResultType]):
    """Base class for all commands, which are requests that expect a response."""

    pass


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


__all__ = ["BaseCommand", "CommandResult", "AddHashesFromStreamCommand", "GetOrCreateEntityFromStreamCommand"]
