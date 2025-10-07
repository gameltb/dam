"""Core components for the command system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from ..enums import ExecutionStrategy
from ..system_events.base import BaseSystemEvent

ResultType = TypeVar("ResultType")
EventType = TypeVar("EventType", bound=BaseSystemEvent)


@dataclass
class BaseCommand[ResultType, EventType: BaseSystemEvent]:
    """Base class for all commands, which are requests that expect a response."""

    execution_strategy: ExecutionStrategy = field(kw_only=True, default=ExecutionStrategy.SERIAL)


@dataclass
class EntityCommand(BaseCommand[ResultType, EventType]):
    """Base class for commands that operate on a single entity."""

    entity_id: int
