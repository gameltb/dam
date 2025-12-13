"""Tests for the DAM API."""

import hashlib
import uuid

import pytest
from dam.core.database import DatabaseManager
from dam.functions.ecs_functions import add_component_to_entity, create_entity
from dam.models.hashes import ContentHashMD5Component
from fastapi.testclient import TestClient

from flowcraft.main import app, lifespan


@pytest.mark.asyncio
async def test_get_entity_components():
    """Test getting all components for an entity."""
    db_name = f"test_db_{uuid.uuid4().hex}"
    async with lifespan(app, db_name=db_name):
        world = app.state.world
        client = TestClient(app)

        db_manager = world.get_resource(DatabaseManager)

        async with db_manager.get_db_session() as session:
            entity = await create_entity(session)
            hash_value = hashlib.md5(b"test_hash").digest()
            await add_component_to_entity(
                session,
                entity.id,
                ContentHashMD5Component(hash_value=hash_value),
            )
            await session.commit()

            response = client.get(f"/entity/{entity.id}")
            assert response.status_code == 200
            data = response.json()
            assert "ContentHashMD5Component" in data
            assert data["ContentHashMD5Component"][0]["hash_value"] == hash_value.hex()
