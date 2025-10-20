"""Configuration for DAM application tests."""

from collections.abc import AsyncGenerator

import pytest_asyncio
from dam.core import DBSession
from dam.core.database import DatabaseManager
from dam_test_utils.types import WorldFactory

pytest_plugins = ["dam_test_utils.fixtures"]


@pytest_asyncio.fixture
async def db_session(world_factory: WorldFactory) -> AsyncGenerator[DBSession, None]:
    """Provide a database session for tests."""
    world = await world_factory("test_world", [])
    async with world.get_resource(DatabaseManager).get_db_session() as session:
        yield session
