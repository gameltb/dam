"""Core Trait class for the DAM system."""

from typing import Any

from dam.core.types import AnySystem
from dam.traits.identifier import TraitIdentifier, TraitImplementationIdentifier


class Trait:
    """Base class for all traits."""

    identifier: TraitIdentifier = TraitIdentifier(parts=("trait",))
    description: str = "A base trait."


class TraitImplementation:
    """A concrete implementation of a trait."""

    def __init__(
        self,
        trait: type[Trait],
        handlers: dict[type[Any], AnySystem],
        identifier: TraitImplementationIdentifier,
        name: str,
        description: str,
    ):
        """Initialize the TraitImplementation."""
        self.trait = trait
        self.handlers = handlers
        self.identifier = identifier
        self.name = name
        self.description = description
