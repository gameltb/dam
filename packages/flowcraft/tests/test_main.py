from fastapi.testclient import TestClient
import pytest
import uuid

@pytest.mark.asyncio
async def test_websocket():
    from flowcraft.main import app, lifespan
    db_name = f"test_db_{uuid.uuid4().hex}"
    async with lifespan(app, db_name=db_name):
        client = TestClient(app)
        with client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            assert data == {"nodes": [], "edges": []}
