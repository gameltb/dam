"""Trait Identifier for the DAM system."""

from __future__ import annotations

import re
from dataclasses import dataclass


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
