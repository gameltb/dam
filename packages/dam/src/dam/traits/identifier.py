"""Trait Identifier for the DAM system."""

from __future__ import annotations

import re
from dataclasses import dataclass

from dam.core.types import ComponentClass


@dataclass(frozen=True)
class TraitIdentifier:
    """A structured identifier for a trait."""

    parts: tuple[str, ...]

    def __post_init__(self):
        """Validate the identifier parts."""
        if not self.parts:
            raise ValueError("Trait identifier cannot be empty.")
        for part in self.parts:
            if not re.match(r"^[a-z0-9_]+$", part):
                raise ValueError(
                    f"Invalid part '{part}' in trait identifier. "
                    "Only lowercase letters, numbers, and underscores are allowed."
                )

    def __str__(self) -> str:
        """Return the string representation of the identifier."""
        return ".".join(self.parts)

    @classmethod
    def from_string(cls, identifier: str) -> TraitIdentifier:
        """Create a TraitIdentifier from a string."""
        if not identifier:
            raise ValueError("Trait identifier cannot be empty.")
        return cls(parts=tuple(identifier.split(".")))


@dataclass(frozen=True)
class TraitImplementationIdentifier:
    """A structured identifier for a trait implementation."""

    trait_id: TraitIdentifier
    component_type: str

    def __str__(self) -> str:
        """Return the string representation of the identifier."""
        return f"{self.trait_id}|{self.component_type}"

    @classmethod
    def from_string(cls, identifier: str) -> TraitImplementationIdentifier:
        """Create a TraitImplementationIdentifier from a string."""
        if "|" not in identifier:
            raise ValueError("Invalid TraitImplementationIdentifier format. Expected 'trait_id|component_type'.")
        trait_id_str, component_type = identifier.split("|", 1)
        return cls(
            trait_id=TraitIdentifier.from_string(trait_id_str),
            component_type=component_type,
        )

    @classmethod
    def from_trait_and_component(
        cls,
        trait_id: TraitIdentifier,
        component: ComponentClass | tuple[ComponentClass, ...],
    ) -> TraitImplementationIdentifier:
        """Create a TraitImplementationIdentifier from a trait and a component."""
        component_type = ",".join(c.__name__ for c in component) if isinstance(component, tuple) else component.__name__
        return cls(trait_id=trait_id, component_type=component_type)
