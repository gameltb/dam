"""Main application file for Flowcraft."""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from dam.core.database import DatabaseManager
from dam.core.world import World
from dam.core.world_manager import create_world_from_components
from dam.functions.ecs_functions import get_all_components_for_entity_as_dict
from dam.models.core.base_class import Base
from dam.plugins.core import CoreSettingsComponent
from fastapi import FastAPI, HTTPException, Request, WebSocket
from psycopg import sql
from pydantic import BaseModel

#
# Database setup
#
db_user = os.environ.get("POSTGRES_USER", "postgres")
db_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
db_host = os.environ.get("POSTGRES_HOST", "localhost")
db_port = os.environ.get("POSTGRES_PORT", "5432")


async def create_temp_db(db_name: str):
    """Create a temporary database for the application."""
    conn = await psycopg.AsyncConnection.connect(
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres", autocommit=True
    )
    try:
        await conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    finally:
        await conn.close()


async def drop_temp_db(db_name: str):
    """Drop the temporary database."""
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


@asynccontextmanager
async def lifespan(_app: FastAPI, db_name: str = f"flowcraft_db_{uuid.uuid4().hex}"):
    """Manage the application's lifespan."""
    db_url = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Setup DAM world
    core_settings = CoreSettingsComponent(
        plugin_name="core",
        database_url=db_url,
        alembic_path="",
    )
    world = create_world_from_components("flowcraft_world", [core_settings])
    app.state.world = world

    await create_temp_db(db_name)
    db_manager = world.get_resource(DatabaseManager)
    await db_manager.create_db_and_tables()
    yield
    if db_manager and db_manager.engine:
        async with db_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await db_manager.engine.dispose()
    await drop_temp_db(db_name)


app = FastAPI(lifespan=lifespan)


class Node(BaseModel):
    """A node in the graph."""

    id: str
    type: str
    position: dict[str, float]
    data: dict[str, Any]


class Edge(BaseModel):
    """An edge in the graph."""

    id: str
    source: str
    target: str


class Graph(BaseModel):
    """The graph."""

    nodes: list[Node]
    edges: list[Edge]


class AppState:
    """The application state."""

    def __init__(self):
        """Initialize the application state."""
        self.graph_store = Graph(nodes=[], edges=[])
        self.connections: list[WebSocket] = []


app_state = AppState()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Manage the websocket endpoint."""
    await websocket.accept()
    app_state.connections.append(websocket)
    try:
        # Send the initial graph state to the newly connected client
        await websocket.send_json(app_state.graph_store.model_dump())
        while True:
            data = await websocket.receive_json()
            # Update the in-memory store with the new graph state.
            app_state.graph_store = Graph.model_validate(data)
            # Broadcast the updated data to all clients
            for connection in app_state.connections:
                await connection.send_json(app_state.graph_store.model_dump())
    except Exception:
        pass
    finally:
        app_state.connections.remove(websocket)


def get_world(request: Request) -> World:
    """Get the world."""
    return request.app.state.world


@app.get("/entity/{entity_id}", response_model=dict[str, list[dict[str, Any]]])
async def get_entity_components(entity_id: int, request: Request):
    """Get all components for an entity."""
    world = get_world(request)
    db_manager = world.get_resource(DatabaseManager)
    async with db_manager.get_db_session() as session:
        components = await get_all_components_for_entity_as_dict(session, entity_id)
        if not components:
            raise HTTPException(status_code=404, detail="Entity not found")
        return components
