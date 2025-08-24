from dataclasses import dataclass, field
from typing import Generic, List, TypeVar

ResultType = TypeVar("ResultType")


@dataclass
class CommandResult(Generic[ResultType]):
    """A container for the results of a command, collecting results from all handlers."""

    results: List[ResultType] = field(default_factory=list)


@dataclass
class BaseCommand(Generic[ResultType]):
    """Base class for all commands, which are requests that expect a response."""

    pass


__all__ = ["BaseCommand", "CommandResult"]
