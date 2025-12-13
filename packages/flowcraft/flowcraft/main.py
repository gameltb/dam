from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
import asyncio
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import os
import uuid
import psycopg
from psycopg import sql

from dam.core.world_manager import create_world_from_components
from dam.models.config import ConfigComponent
from dam.plugins.core import CoreSettingsComponent
from dam.core.database import DatabaseManager
from dam.functions.ecs_functions import get_all_components_for_entity_as_dict
from dam.models.core.base_class import Base

#
# Database setup
#
db_user = os.environ.get("POSTGRES_USER", "postgres")
db_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
db_host = os.environ.get("POSTGRES_HOST", "localhost")
db_port = os.environ.get("POSTGRES_PORT", "5432")

async def create_temp_db(db_name: str):
    conn = await psycopg.AsyncConnection.connect(
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres", autocommit=True
    )
    try:
        await conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    finally:
        await conn.close()

async def drop_temp_db(db_name: str):
    conn = await psycopg.AsyncConnection.connect(
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres", autocommit=True
    )
    try:
        # Terminate all connections to the database to allow it to be dropped.
        await conn.execute(
            sql.SQL("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s"), (db_name,)
        )
        await conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name)))
    except psycopg.errors.InvalidCatalogName:
        pass  # Database already dropped
    finally:
        await conn.close()

world = None
app = FastAPI() # Create app instance here

@asynccontextmanager
async def lifespan(app: FastAPI, db_name: str = f"flowcraft_db_{uuid.uuid4().hex}"):
    global world
    db_url = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Setup DAM world
    core_settings = CoreSettingsComponent(
        plugin_name="core",
        database_url=db_url,
        alembic_path="",
    )
    world = create_world_from_components("flowcraft_world", [core_settings])

    await create_temp_db(db_name)
    db_manager = world.get_resource(DatabaseManager)
    await db_manager.create_db_and_tables()
    yield
    if db_manager and db_manager.engine:
        async with db_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await db_manager.engine.dispose()
    await drop_temp_db(db_name)

# This is a bit of a hack to make the lifespan function work with the tests
app.router.lifespan_context = lifespan

class Node(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: Dict

class Edge(BaseModel):
    id: str
    source: str
    target: str

class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

# In-memory store for the graph
graph_store = Graph(nodes=[], edges=[])

# List of active WebSocket connections
connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global graph_store
    await websocket.accept()
    connections.append(websocket)
    try:
        # Send the initial graph state to the newly connected client
        await websocket.send_json(graph_store.model_dump())
        while True:
            data = await websocket.receive_json()
            # Update the in-memory store with the new graph state.
            graph_store = Graph.parse_obj(data)
            # Broadcast the updated data to all clients
            for connection in connections:
                await connection.send_json(graph_store.model_dump())
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connections.remove(websocket)

@app.get("/entity/{entity_id}", response_model=Dict[str, List[Dict[str, Any]]])
async def get_entity_components(entity_id: int):
    db_manager = world.get_resource(DatabaseManager)
    async with db_manager.get_db_session() as session:
        components = await get_all_components_for_entity_as_dict(session, entity_id)
        if not components:
            raise HTTPException(status_code=404, detail="Entity not found")
        return components
