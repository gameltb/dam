"""Data classes for storing system metadata."""

# pyright: basic
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dam.commands.core import BaseCommand
from dam.core.enums import SystemType
from dam.core.stages import SystemStage
from dam.events import BaseEvent
from dam.models.core.base_component import BaseComponent


@dataclass
class SystemParameterInfo:
    """Holds metadata about a single parameter of a system function."""

    name: str
    type_hint: Any
    identity: type[Any] | None
    marker_component_type: type[BaseComponent] | None
    event_type_hint: type[BaseEvent] | None
    command_type_hint: type[BaseCommand[Any, Any]] | None
    is_annotated: bool
    original_annotation: Any


@dataclass
class SystemMetadata:
    """Holds metadata about a system function."""

    func: Callable[..., Any]
    params: dict[str, SystemParameterInfo]
    is_async: bool
    system_type: SystemType
    stage: SystemStage | None
    handles_command_type: type[BaseCommand[Any, Any]] | None
    listens_for_event_type: type[BaseEvent] | None
