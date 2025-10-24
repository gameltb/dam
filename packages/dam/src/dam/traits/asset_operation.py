"""A trait for asset operations."""

from dataclasses import dataclass

from dam.commands import asset_operation_commands
from dam.traits.identifier import TraitIdentifier
from dam.traits.traits import Trait


@dataclass
class AssetOperationTrait(Trait):
    """A trait for components that represent an asset operation."""

    identifier = TraitIdentifier.from_string("asset.operation")
    description = "Provides a way to perform an operation on an asset."

    Add = asset_operation_commands.Add
    Check = asset_operation_commands.Check
    Remove = asset_operation_commands.Remove
    ReprocessDerived = asset_operation_commands.ReprocessDerived
