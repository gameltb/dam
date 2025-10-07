"""Core transaction management for the DAM system."""

import logging
from contextvars import ContextVar
from typing import Any, Optional, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from dam.functions import ecs_functions
from dam.models.core.base_component import Component, UniqueComponent
from dam.models.core.entity import Entity

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Component)


class WorldTransaction:
    """
    A wrapper around the SQLAlchemy session and ECS functions.

    This provides a controlled, transactional interface for systems.
    """

    def __init__(self, session: AsyncSession):
        """Initialize the transaction with an active session."""
        self.session = session

    async def flush(self) -> None:
        """Flush the underlying session to persist changes within the current transaction."""
        await self.session.flush()

    async def create_entity(self) -> Entity:
        """Create a new entity."""
        return await ecs_functions.create_entity(self.session)

    async def get_entity(self, entity_id: int) -> Entity | None:
        """Get an entity by its ID."""
        return await ecs_functions.get_entity(self.session, entity_id)

    async def add_component_to_entity(self, entity_id: int, component_instance: T) -> T:
        """Add a component to an entity."""
        # Note: The underlying functions has a `flush` parameter which defaults to True.
        # In this pattern, we want to control the flush at a higher level, so we set it to False.
        return await ecs_functions.add_component_to_entity(self.session, entity_id, component_instance, flush=False)

    async def get_component(self, entity_id: int, component_type: type[T]) -> T | None:
        """Get a single component of a specific type for an entity."""
        return await ecs_functions.get_component(self.session, entity_id, component_type)

    async def get_components(self, entity_id: int, component_type: type[T]) -> list[T]:
        """Get all components of a specific type for an entity."""
        return await ecs_functions.get_components(self.session, entity_id, component_type)

    async def get_all_components_for_entity(self, entity_id: int) -> list[Component]:
        """Get all component instances associated with a given entity."""
        return await ecs_functions.get_all_components_for_entity(self.session, entity_id)

    async def add_or_update_component(self, entity_id: int, component_instance: T) -> T:
        """
        Add a component to an entity.

        If the component is a UniqueComponent and one already exists,
        it updates the existing component with the new values.
        """
        if isinstance(component_instance, UniqueComponent):
            existing_component = await self.get_component(entity_id, type(component_instance))
            if existing_component:
                # Update existing component
                for key, value in component_instance.__dict__.items():
                    if key.startswith("_"):
                        continue
                    setattr(existing_component, key, value)
                self.session.add(existing_component)
                return existing_component

        # If it's not a UniqueComponent or doesn't exist yet, add it.
        return await self.add_component_to_entity(entity_id, component_instance)

    async def remove_component(self, component: Component) -> None:
        """Remove a component from an entity."""
        return await ecs_functions.remove_component(self.session, component, flush=False)

    async def delete_entity(self, entity_id: int) -> bool:
        """Delete an entity and all its components."""
        return await ecs_functions.delete_entity(self.session, entity_id, flush=False)

    async def find_entity_by_content_hash(self, hash_value: bytes, hash_type: str = "sha256") -> Entity | None:
        """Find an entity by its content hash."""
        return await ecs_functions.find_entity_by_content_hash(self.session, hash_value, hash_type)

    async def get_components_by_value(
        self, entity_id: int, component_type: type[T], attributes_values: dict[str, Any]
    ) -> list[T]:
        """Get components of a specific type for an entity that match all given attribute values."""
        return await ecs_functions.get_components_by_value(self.session, entity_id, component_type, attributes_values)

    async def find_entities_by_component_attribute_value(
        self, component_type: type[T], attribute_name: str, value: Any
    ) -> list[Entity]:
        """Find entities that have a component with a specific attribute value."""
        return await ecs_functions.find_entities_by_component_attribute_value(
            self.session, component_type, attribute_name, value
        )


# Context variable to hold the current WorldTransaction instance for a given async context.
active_transaction: ContextVar[Optional["WorldTransaction"]] = ContextVar("active_transaction", default=None)
