"""Teardown functions for DAM tests."""

from dam.core.database import DatabaseManager
from dam.core.world import World
from dam.models.core.base_class import Base


async def teardown_world_async(world: World) -> None:
    """Teardown a world after a test."""
    if world and world.has_resource(DatabaseManager):
        db_mngr = world.get_resource(DatabaseManager)
        if db_mngr and db_mngr.engine:
            async with db_mngr.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            await db_mngr.engine.dispose()
