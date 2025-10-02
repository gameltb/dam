from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from dam.system_events.base import BaseSystemEvent

T = TypeVar("T")


@dataclass
class InformationRequest(BaseSystemEvent, Generic[T]):
    """
    Base class for information requests that can be yielded by systems.
    This allows a system to pause its execution, request information from the user,
    and then resume with the provided data.
    """

    pass


@dataclass
class PasswordRequest(InformationRequest[Optional[str]]):
    """
    A specific information request for a password.
    """

    message: str = "Password required"
