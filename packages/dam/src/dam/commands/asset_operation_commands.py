"""Generic commands for asset operations."""

from dataclasses import dataclass
from typing import Any

from dam.commands.core import EntityCommand
from dam.system_events.base import BaseSystemEvent


@dataclass
class Add(EntityCommand[Any, Any]):
    """Abstract command to add the operation to an asset."""


@dataclass
class Check(EntityCommand[bool, BaseSystemEvent]):
    """Abstract command to check if the operation has been applied to an asset."""


@dataclass
class Remove(EntityCommand[None, BaseSystemEvent]):
    """Abstract command to remove the operation from an asset."""


@dataclass
class ReprocessDerived(EntityCommand[None, BaseSystemEvent]):
    """Abstract command to reprocess derived assets."""
