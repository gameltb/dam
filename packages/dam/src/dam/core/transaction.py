import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Type, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from dam.functions import ecs_functions
from dam.models.core.base_component import Component, UniqueComponent
from dam.models.core.entity import Entity

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Component)


class WorldTransaction:
    """
    A wrapper around the SQLAlchemy session and ECS functions that provides
    a controlled, transactional interface for systems.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def flush(self) -> None:
        """Flushes the underlying session to persist changes within the current transaction."""
        await self.session.flush()

    async def create_entity(self) -> Entity:
        return await ecs_functions.create_entity(self.session)

    async def get_entity(self, entity_id: int) -> Optional[Entity]:
        return await ecs_functions.get_entity(self.session, entity_id)

    async def add_component_to_entity(self, entity_id: int, component_instance: T) -> T:
        # Note: The underlying functions has a `flush` parameter which defaults to True.
        # In this pattern, we want to control the flush at a higher level, so we set it to False.
        return await ecs_functions.add_component_to_entity(self.session, entity_id, component_instance, flush=False)

    async def get_component(self, entity_id: int, component_type: Type[T]) -> Optional[T]:
        return await ecs_functions.get_component(self.session, entity_id, component_type)

    async def get_components(self, entity_id: int, component_type: Type[T]) -> List[T]:
        return await ecs_functions.get_components(self.session, entity_id, component_type)

    async def get_all_components_for_entity(self, entity_id: int) -> List[Component]:
        return await ecs_functions.get_all_components_for_entity(self.session, entity_id)

    async def add_or_update_component(self, entity_id: int, component_instance: T) -> T:
        """
        Adds a component to an entity. If the component is a UniqueComponent and one
        already exists, it updates the existing component with the new values.
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
        return await ecs_functions.remove_component(self.session, component, flush=False)

    async def delete_entity(self, entity_id: int) -> bool:
        return await ecs_functions.delete_entity(self.session, entity_id, flush=False)

    async def find_entity_by_content_hash(self, hash_value: bytes, hash_type: str = "sha256") -> Optional[Entity]:
        return await ecs_functions.find_entity_by_content_hash(self.session, hash_value, hash_type)

    async def get_components_by_value(
        self, entity_id: int, component_type: Type[T], attributes_values: Dict[str, Any]
    ) -> List[T]:
        return await ecs_functions.get_components_by_value(self.session, entity_id, component_type, attributes_values)

    async def find_entities_by_component_attribute_value(
        self, component_type: Type[T], attribute_name: str, value: Any
    ) -> List[Entity]:
        return await ecs_functions.find_entities_by_component_attribute_value(
            self.session, component_type, attribute_name, value
        )


# Context variable to hold the current WorldTransaction instance for a given async context.
active_transaction: ContextVar[Optional["WorldTransaction"]] = ContextVar("active_transaction", default=None)
