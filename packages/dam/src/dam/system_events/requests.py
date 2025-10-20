"""System events for requesting information from the user."""

import asyncio
from dataclasses import dataclass, field
from typing import TypeVar

from dam.system_events.base import BaseSystemEvent

T = TypeVar("T")


@dataclass
class InformationRequest[T](BaseSystemEvent):
    """
    Base class for information requests that can be yielded by systems.

    This allows a system to pause its execution, request information from the user,
    and then resume with the provided data.
    """

    future: asyncio.Future[T] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the future with the correct type."""
        self.future = asyncio.Future()


@dataclass
class PasswordRequest(InformationRequest[str | None]):
    """A specific information request for a password."""

    message: str = "Password required"
