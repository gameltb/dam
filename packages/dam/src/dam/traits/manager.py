"""Core TraitManager class for the DAM system."""

from collections import defaultdict
from typing import Any

from dam.core.types import AnySystem
from dam.models import BaseComponent
from dam.traits.traits import Trait


class TraitImplementation:
    """A concrete implementation of a trait."""

    def __init__(self, trait: type[Trait], handlers: dict[type[Any], AnySystem]):
        """Initialize the TraitImplementation."""
        self.trait = trait
        self.handlers = handlers


class TraitManager:
    """Manages the registration and discovery of traits."""

    def __init__(self) -> None:
        """Initialize the TraitManager."""
        self._implementations: dict[
            type[BaseComponent] | tuple[type[BaseComponent], ...], list[TraitImplementation]
        ] = defaultdict(list)
        self._trait_map: dict[str, type[Trait]] = {}

    def register(
        self,
        component_type: type[BaseComponent] | tuple[type[BaseComponent], ...],
        implementation: TraitImplementation,
    ) -> None:
        """Register a trait implementation for a component type."""
        identifier = str(implementation.trait.identifier)
        if identifier in self._trait_map and self._trait_map[identifier] is not implementation.trait:
            raise ValueError(f"Trait with identifier '{identifier}' already registered.")

        self._trait_map[identifier] = implementation.trait
        self._implementations[component_type].append(implementation)

    def get_implementations_for_components(
        self, component_types: set[type[BaseComponent]]
    ) -> list[TraitImplementation]:
        """Get all trait implementations for a set of component types."""
        implementations: list[TraitImplementation] = []
        for key, impls in self._implementations.items():
            if isinstance(key, tuple):
                if all(k in component_types for k in key):
                    implementations.extend(impls)
            elif key in component_types:
                implementations.extend(impls)
        return implementations

    def get_trait_handlers(self, component_types: set[type[BaseComponent]]) -> dict[type[Any], AnySystem]:
        """Get all trait handlers for a set of component types."""
        handlers: dict[type[Any], AnySystem] = {}
        implementations = self.get_implementations_for_components(component_types)
        for impl in implementations:
            handlers.update(impl.handlers)
        return handlers

    def get_trait_by_id(self, identifier: str) -> type[Trait] | None:
        """Get a trait by its identifier."""
        return self._trait_map.get(identifier)
