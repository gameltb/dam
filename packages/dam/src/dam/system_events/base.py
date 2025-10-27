"""Base classes for system events."""

from typing import TypeVar

T = TypeVar("T")


class BaseSystemEvent:
    """A base class for all events yielded by systems."""

    pass


class SystemResultEvent[T](BaseSystemEvent):
    """An event that wraps the final return value of a non-generator system."""

    def __init__(self, result: T):
        """Initialize the event with the result."""
        self.result = result


__all__ = ["BaseSystemEvent", "SystemResultEvent"]
