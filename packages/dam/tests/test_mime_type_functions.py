"""Tests for MIME type functions."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from dam_test_utils.types import WorldFactory
from sqlalchemy.ext.asyncio import AsyncSession

from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.functions import ecs_functions
from dam.functions import mime_type_functions as mt_funks
from dam.models.core.entity import Entity


@pytest_asyncio.fixture
async def db_session(world_factory: WorldFactory) -> AsyncGenerator[AsyncSession, None]:
    """Create a new database session for a test."""
    world: World = await world_factory("test_world", [])
    async with world.get_context(WorldTransaction)() as tx:
        yield tx.session


@pytest_asyncio.fixture
async def generic_entity(db_session: AsyncSession) -> Entity:
    """Provide a generic entity for testing."""
    entity = await ecs_functions.create_entity(db_session)
    assert entity is not None
    return entity


@pytest.mark.asyncio
async def test_get_or_create_mime_type_concept(db_session: AsyncSession):
    """Test getting or creating a MIME type concept."""
    # Test creation
    mime_type = "image/jpeg"
    concept_comp = await mt_funks.get_or_create_mime_type_concept(db_session, mime_type)
    assert concept_comp is not None
    assert concept_comp.mime_type == mime_type

    # Test retrieval
    concept_comp_2 = await mt_funks.get_or_create_mime_type_concept(db_session, mime_type)
    assert concept_comp_2 is not None
    assert concept_comp_2.id == concept_comp.id

    # Test empty name
    assert await mt_funks.get_or_create_mime_type_concept(db_session, "") is None


@pytest.mark.asyncio
async def test_set_and_get_content_mime_type(db_session: AsyncSession, generic_entity: Entity):
    """Test setting and getting the content MIME type for an entity."""
    mime_type = "application/pdf"

    # Set the mime type
    content_mime_comp = await mt_funks.set_content_mime_type(db_session, generic_entity.id, mime_type)
    assert content_mime_comp is not None
    assert content_mime_comp.entity_id == generic_entity.id

    # Get the mime type
    retrieved_mime_type = await mt_funks.get_content_mime_type(db_session, generic_entity.id)
    assert retrieved_mime_type == mime_type

    # Test getting mime type for entity that doesn't have one
    new_entity = await ecs_functions.create_entity(db_session)
    assert await mt_funks.get_content_mime_type(db_session, new_entity.id) is None


@pytest.mark.asyncio
async def test_update_content_mime_type(db_session: AsyncSession, generic_entity: Entity):
    """Test updating the content MIME type for an entity."""
    # Set initial mime type
    await mt_funks.set_content_mime_type(db_session, generic_entity.id, "image/png")

    # Update to a new mime type
    await mt_funks.set_content_mime_type(db_session, generic_entity.id, "image/gif")

    retrieved_mime_type = await mt_funks.get_content_mime_type(db_session, generic_entity.id)
    assert retrieved_mime_type == "image/gif"
