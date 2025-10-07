"""System events for requesting information from the user."""

from dataclasses import dataclass
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

    pass


@dataclass
class PasswordRequest(InformationRequest[str | None]):
    """A specific information request for a password."""

    message: str = "Password required"
