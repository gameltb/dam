from fastapi.testclient import TestClient
from dam.functions.ecs_functions import create_entity, add_component_to_entity
from dam.models.hashes import ContentHashMD5Component
from dam.core.database import DatabaseManager
import pytest
import hashlib
import uuid

# Since the app is defined in another test file, we can't import it here.
# Instead, we'll create a new TestClient instance in the test function.
# This is not ideal, but it's the simplest solution for now.

@pytest.mark.asyncio
async def test_get_entity_components():
    from flowcraft.main import app, lifespan  # Import here to avoid circular dependencies

    db_name = f"test_db_{uuid.uuid4().hex}"
    async with lifespan(app, db_name=db_name):
        from flowcraft.main import world # world is now available
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
