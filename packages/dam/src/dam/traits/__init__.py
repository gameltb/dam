"""Core Trait classes for the DAM system."""

from .manager import TraitImplementation, TraitManager
from .traits import Trait

__all__ = [
    "Trait",
    "TraitImplementation",
    "TraitManager",
]
