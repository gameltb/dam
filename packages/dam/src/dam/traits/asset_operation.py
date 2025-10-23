"""A trait for asset operations."""

from dataclasses import dataclass
from typing import Any

from dam.commands.core import EntityCommand
from dam.system_events.base import BaseSystemEvent
from dam.traits.identifier import TraitIdentifier
from dam.traits.traits import Trait


class AssetOperationTrait(Trait):
    """A trait for components that represent an asset operation."""

    identifier = TraitIdentifier.from_string("asset.operation")
    description = "Provides a way to perform an operation on an asset."

    @dataclass
    class Add(EntityCommand[Any, Any]):
        """Abstract command to add the operation to an asset."""

    @dataclass
    class Check(EntityCommand[bool, BaseSystemEvent]):
        """Abstract command to check if the operation has been applied to an asset."""

    @dataclass
    class Remove(EntityCommand[None, BaseSystemEvent]):
        """Abstract command to remove the operation from an asset."""
