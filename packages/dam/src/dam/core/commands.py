from dataclasses import dataclass, field
from typing import Any, Generic, List, TypeVar

T = TypeVar("T")


@dataclass
class CommandResult(Generic[T]):
    """A container for the results of a command, collecting results from all handlers."""

    results: List[T] = field(default_factory=list)


@dataclass
class BaseCommand:
    """Base class for all commands, which are requests that expect a response."""

    pass


__all__ = ["BaseCommand", "CommandResult"]
