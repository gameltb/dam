from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator

from dam.core.config import WorldConfig
from dam.core.contexts import ContextProvider
from dam.core.database import DatabaseManager
from dam.core.transaction import WorldTransaction, active_transaction

if TYPE_CHECKING:
    pass


class TransactionManager(ContextProvider[WorldTransaction]):
    def __init__(self, world_config: WorldConfig):
        self.db_manager = DatabaseManager(world_config)

    async def create_db_and_tables(self) -> None:
        await self.db_manager.create_db_and_tables()

    @asynccontextmanager
    async def __call__(self, **kwargs: Any) -> AsyncGenerator[WorldTransaction, None]:
        transaction = active_transaction.get()
        use_nested_transaction = kwargs.get("use_nested_transaction", False)

        if transaction:
            if use_nested_transaction:
                async with transaction.session.begin_nested():
                    yield transaction
            else:
                yield transaction
            return

        # If there's no active transaction, create a new top-level one.
        db_session = self.db_manager.get_db_session()
        try:
            async with db_session.begin():
                new_transaction = WorldTransaction(db_session)
                token = active_transaction.set(new_transaction)
                try:
                    yield new_transaction
                finally:
                    active_transaction.reset(token)
        finally:
            await db_session.close()
