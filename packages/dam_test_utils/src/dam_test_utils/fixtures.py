"""Fixtures for DAM tests."""

import os
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import (
    Any,
)

import numpy as np
import psycopg
import pytest
import pytest_asyncio
import torch
from dam import world_manager
from dam.core import plugin_loader
from dam.core.config_loader import load_and_validate_settings
from dam.core.database import DatabaseManager
from dam.core.world import World
from dam.models.core.base_class import Base
from PIL import Image
from psycopg import sql
from sqlalchemy.ext.asyncio import AsyncSession

MOCK_TRANSFORMER_EMBEDDING_DIM = 3


@pytest_asyncio.fixture(scope="function")
async def test_db() -> AsyncGenerator[str, None]:
    """Create a temporary database for testing."""
    db_user = os.environ.get("POSTGRES_USER", "postgres")
    db_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    db_host = os.environ.get("POSTGRES_HOST", "localhost")
    db_port = os.environ.get("POSTGRES_PORT", "5432")
    db_name = f"test_db_{uuid.uuid4().hex}"

    conn = await psycopg.AsyncConnection.connect(
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres", autocommit=True
    )
    try:
        await conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    finally:
        await conn.close()

    db_url = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    yield db_url

    conn = await psycopg.AsyncConnection.connect(
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres", autocommit=True
    )
    try:
        await conn.execute(
            sql.SQL("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s"), (db_name,)
        )
        await conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name)))
    except psycopg.errors.InvalidCatalogName:
        pass
    finally:
        await conn.close()


@pytest.fixture
def test_toml_config(test_db: str, tmp_path: Path) -> Path:
    """Creates a temporary dam.toml file for testing."""
    worlds_config = ""
    world_names = [
        "test_world_alpha",
        "test_world_beta",
        "test_world_gamma",
    ]

    for name in world_names:
        asset_storage_path = (tmp_path / f"assets_{name}").as_posix().replace("\\", "/")
        worlds_config += f"""
[worlds.{name}.plugin_settings.core]
DATABASE_URL = "{test_db}"

[worlds.{name}.plugin_settings."dam-fs"]
ASSET_STORAGE_PATH = "{asset_storage_path}"
"""

    toml_path = tmp_path / "dam.toml"
    toml_path.write_text(worlds_config)
    return toml_path


async def _setup_world(world_name: str, test_toml_config: Path, plugins: list[Any] | None = None) -> World:
    """Set up a dam world, with optional plugins."""
    loaded_components = load_and_validate_settings(test_toml_config)

    world = World(name=world_name)
    world.add_resource(world, World)

    world_settings = loaded_components.get(world_name, {})
    for component in world_settings.values():
        world.add_resource(component, component.__class__)

    for plugin_name in world_settings:
        plugin = plugin_loader.load_plugin(plugin_name)
        if plugin:
            world.add_plugin(plugin)

    if plugins:
        for plugin in plugins:
            world.add_plugin(plugin)

    db_manager = world.get_resource(DatabaseManager)
    await db_manager.create_db_and_tables()

    world_manager.register_world(world)
    return world


async def _teardown_world_async(world: World) -> None:
    if world and world.has_resource(DatabaseManager):
        db_mngr = world.get_resource(DatabaseManager)
        if db_mngr and db_mngr.engine:
            async with db_mngr.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            await db_mngr.engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_world_alpha(test_toml_config: Path) -> AsyncGenerator[World, None]:
    """Create a test world named 'alpha'."""
    world = await _setup_world("test_world_alpha", test_toml_config, plugins=None)
    yield world
    await _teardown_world_async(world)


@pytest_asyncio.fixture(scope="function")
async def db_session(test_world_alpha: World) -> AsyncGenerator[AsyncSession, None]:
    """Create a new database session for a test."""
    db_mngr = test_world_alpha.get_resource(DatabaseManager)
    async with db_mngr.session_local() as session:
        yield session


class MockSentenceTransformer(torch.nn.Module):
    """A mock sentence transformer for testing."""

    def __init__(self, model_name_or_path: str | None = None, **_kwargs: Any) -> None:
        """Initialize the mock sentence transformer."""
        super().__init__()
        self.model_name = model_name_or_path
        self.dim = 384

    def encode(self, sentences: str | list[str], **_kwargs: Any) -> np.ndarray:
        """Encode sentences into embeddings."""
        if isinstance(sentences, str):
            return np.zeros(self.dim, dtype=np.float32)
        return np.zeros((len(sentences), self.dim), dtype=np.float32)


@pytest_asyncio.fixture(scope="function")
async def test_world_beta(test_toml_config: Path) -> AsyncGenerator[World, None]:
    """Create a test world named 'beta'."""
    world = await _setup_world("test_world_beta", test_toml_config)
    yield world
    await _teardown_world_async(world)


@pytest_asyncio.fixture(scope="function")
async def test_world_gamma(test_toml_config: Path) -> AsyncGenerator[World, None]:
    """Create a test world named 'gamma'."""
    world = await _setup_world("test_world_gamma", test_toml_config)
    yield world
    await _teardown_world_async(world)


@pytest.fixture
def temp_asset_file(tmp_path: Path) -> Path:
    """Create a temporary asset file."""
    file_path = tmp_path / "test_asset.txt"
    file_path.write_text("This is a test asset.")
    return file_path


@pytest.fixture
def temp_image_file(tmp_path: Path) -> Path:
    """Create a temporary image file."""
    file_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (60, 30), color="red")
    img.save(file_path)
    return file_path
