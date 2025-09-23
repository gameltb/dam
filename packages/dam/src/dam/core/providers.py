# pyright: basic

from typing import List, Type, AsyncGenerator
from contextlib import asynccontextmanager

from dam.core.transaction import WorldTransaction
from dam.models.core.base_component import BaseComponent
from dam.models.core.entity import Entity

from dam.core.contexts import ContextProvider


class MarkedEntityListProvider(ContextProvider[List[Entity]]):
    """
    Provides a list of entities that are marked with a specific component.
    """

    @asynccontextmanager
    async def __call__(
        self,
        marker_component_type: Type[BaseComponent],
        transaction: WorldTransaction,
    ) -> AsyncGenerator[List[Entity], None]:
        """
        Queries for entities with the given marker component within a transaction.
        """
        from sqlalchemy import exists as sql_exists
        from sqlalchemy import select as sql_select

        stmt = sql_select(Entity).where(
            sql_exists().where(marker_component_type.entity_id == Entity.id)
        )
        result = await transaction.session.execute(stmt)
        entities_to_process = list(result.scalars().all())
        yield entities_to_process
