"""Core Trait class for the DAM system."""

from dam.traits.identifier import TraitIdentifier


class Trait:
    """Base class for all traits."""

    identifier: TraitIdentifier = TraitIdentifier(parts=("trait",))
    description: str = "A base trait."
