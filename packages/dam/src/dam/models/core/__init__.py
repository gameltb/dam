"""Core data models for the DAM system."""
# This file makes the 'core' directory a Python package.

# Example:
from .base_component import (
    BaseComponent,
    UniqueComponent,
)  # Assuming BaseComponent should also be exported
from .entity import Entity

__all__ = ["BaseComponent", "Entity", "UniqueComponent"]
