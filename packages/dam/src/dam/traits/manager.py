"""Core TraitManager class for the DAM system."""

from collections import defaultdict
from typing import Any

from dam.core.types import AnySystem, ComponentClass
from dam.models import BaseComponent
from dam.traits.identifier import TraitImplementationIdentifier
from dam.traits.traits import Trait, TraitImplementation


class TraitManager:
    """Manages the registration and discovery of traits."""

    def __init__(self) -> None:
        """Initialize the TraitManager."""
        self._implementations_by_component: dict[
            ComponentClass | tuple[ComponentClass, ...], list[TraitImplementation]
        ] = defaultdict(list)
        self._implementations_by_id: dict[TraitImplementationIdentifier, TraitImplementation] = {}
        self._trait_map: dict[str, type[Trait]] = {}

    def register(
        self,
        component_type: ComponentClass | tuple[ComponentClass, ...],
        implementation: TraitImplementation,
    ) -> None:
        """Register a trait implementation for a component type."""
        trait_id = str(implementation.trait.identifier)
        if trait_id in self._trait_map and self._trait_map[trait_id] is not implementation.trait:
            raise ValueError(f"Trait with identifier '{trait_id}' already registered.")

        self._trait_map[trait_id] = implementation.trait
        self._implementations_by_component[component_type].append(implementation)
        self._implementations_by_id[implementation.identifier] = implementation

    def get_implementations_for_components(
        self, component_types: set[type[BaseComponent]]
    ) -> list[TraitImplementation]:
        """Get all trait implementations for a set of component types."""
        implementations: list[TraitImplementation] = []
        for key, impls in self._implementations_by_component.items():
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

    def get_implementation_by_id(self, identifier: TraitImplementationIdentifier) -> TraitImplementation | None:
        """Get a trait implementation by its identifier."""
        return self._implementations_by_id.get(identifier)

    def get_implementations_for_trait(self, trait: type[Trait]) -> list[TraitImplementation]:
        """Get all implementations for a trait."""
        return [impl for impl in self._implementations_by_id.values() if impl.trait is trait]
