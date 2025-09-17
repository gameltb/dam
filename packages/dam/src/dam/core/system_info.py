# pyright: basic
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from dam.core.commands import BaseCommand
from dam.core.enums import SystemType
from dam.core.events import BaseEvent
from dam.core.stages import SystemStage
from dam.models.core.base_component import BaseComponent


@dataclass
class SystemParameterInfo:
    """
    Holds metadata about a single parameter of a system function.
    """

    name: str
    type_hint: Any
    identity: Optional[Type[Any]]
    marker_component_type: Optional[Type[BaseComponent]]
    event_type_hint: Optional[Type[BaseEvent]]
    command_type_hint: Optional[Type[BaseCommand[Any, Any]]]
    is_annotated: bool
    original_annotation: Any


@dataclass
class SystemMetadata:
    """
    Holds metadata about a system function.
    """

    func: Callable[..., Any]
    params: Dict[str, SystemParameterInfo]
    is_async: bool
    system_type: SystemType
    stage: Optional[SystemStage]
    handles_command_type: Optional[Type[BaseCommand[Any, Any]]]
    listens_for_event_type: Optional[Type[BaseEvent]]
