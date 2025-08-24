import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Type, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from dam.models.core.base_component import BaseComponent
from dam.models.core.entity import Entity
from dam.services import ecs_service

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseComponent)


class EcsTransaction:
    """
    A wrapper around the SQLAlchemy session and ECS services that provides
    a controlled, transactional interface for systems.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def flush(self):
        """Flushes the underlying session to persist changes within the current transaction."""
        await self.session.flush()

    async def create_entity(self) -> Entity:
        return await ecs_service.create_entity(self.session)

    async def get_entity(self, entity_id: int) -> Optional[Entity]:
        return await ecs_service.get_entity(self.session, entity_id)

    async def add_component_to_entity(self, entity_id: int, component_instance: T) -> T:
        # Note: The underlying service has a `flush` parameter which defaults to True.
        # In this pattern, we want to control the flush at a higher level, so we set it to False.
        return await ecs_service.add_component_to_entity(self.session, entity_id, component_instance, flush=False)

    async def get_component(self, entity_id: int, component_type: Type[T]) -> Optional[T]:
        return await ecs_service.get_component(self.session, entity_id, component_type)

    async def get_components(self, entity_id: int, component_type: Type[T]) -> List[T]:
        return await ecs_service.get_components(self.session, entity_id, component_type)

    async def get_all_components_for_entity(self, entity_id: int) -> List[BaseComponent]:
        return await ecs_service.get_all_components_for_entity(self.session, entity_id)

    async def remove_component(self, component: BaseComponent):
        return await ecs_service.remove_component(self.session, component, flush=False)

    async def delete_entity(self, entity_id: int) -> bool:
        return await ecs_service.delete_entity(self.session, entity_id, flush=False)

    async def find_entity_by_content_hash(self, hash_value: bytes, hash_type: str = "sha256") -> Optional[Entity]:
        return await ecs_service.find_entity_by_content_hash(self.session, hash_value, hash_type)

    async def get_components_by_value(
        self, entity_id: int, component_type: Type[T], attributes_values: Dict[str, Any]
    ) -> List[T]:
        return await ecs_service.get_components_by_value(self.session, entity_id, component_type, attributes_values)

    async def find_entities_by_component_attribute_value(
        self, component_type: Type[T], attribute_name: str, value: Any
    ) -> List[Entity]:
        return await ecs_service.find_entities_by_component_attribute_value(
            self.session, component_type, attribute_name, value
        )


# Context variable to hold the current EcsTransaction instance for a given async context.
active_transaction: ContextVar[Optional["EcsTransaction"]] = ContextVar("active_transaction", default=None)
