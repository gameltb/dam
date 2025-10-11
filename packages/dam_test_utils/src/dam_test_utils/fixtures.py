"""Fixtures for DAM tests."""

import os
import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import (
    Any,
)

import numpy as np
import psycopg
import pytest
import pytest_asyncio
import tomli_w
import torch
from dam.core.config import Config, DamToml
from dam.core.database import DatabaseManager
from dam.core.world import World
from dam.core.world_manager import create_world, world_manager
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

    conn = psycopg.connect(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres", autocommit=True)
    try:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    finally:
        conn.close()

    db_url = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    yield db_url

    conn = psycopg.connect(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres", autocommit=True)
    try:
        conn.execute(sql.SQL("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s"), (db_name,))
        conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name)))
    except psycopg.errors.InvalidCatalogName:
        pass
    finally:
        conn.close()


@pytest.fixture
def test_config(test_db: str, tmp_path: Path) -> Generator[Config, None, None]:
    """Creates a temporary dam.toml and returns a parsed Config object."""

    def _factory() -> dict[str, dict[str, str]]:
        return {
            "test_world_alpha": {"DATABASE_URL": test_db},
            "test_world_beta": {"DATABASE_URL": test_db},
            "test_world_gamma": {"DATABASE_URL": test_db},
        }

    raw_world_configs = _factory()
    worlds_config_dict: dict[str, Any] = {}

    for world_name, config_template in raw_world_configs.items():
        asset_temp_dir = tmp_path / f"assets_{world_name}"
        asset_temp_dir.mkdir(parents=True, exist_ok=True)

        worlds_config_dict[world_name] = {
            "db": {"url": config_template["DATABASE_URL"]},
            "plugins": {"names": ["dam-fs"]},
            "paths": {},
            "plugin_settings": {
                "dam-fs": {
                    "storage_path": str(asset_temp_dir),
                }
            },
        }

    final_toml_dict = {"worlds": worlds_config_dict}

    toml_path = tmp_path / "dam.toml"
    with toml_path.open("wb") as f:
        tomli_w.dump(final_toml_dict, f)

    config_loader = DamToml(start_dir=tmp_path)
    config = config_loader.parse()

    world_manager.clear_world_registry()
    yield config
    world_manager.clear_world_registry()


async def _setup_world(world_name: str, test_config_fixture: Config, plugins: list[Any] | None = None) -> World:
    """Set up a dam world, with optional plugins."""
    world = create_world(world_name, config=test_config_fixture)
    world_manager.register_world(world)

    db_manager = world.get_resource(DatabaseManager)
    await db_manager.create_db_and_tables()

    if plugins:
        for plugin in plugins:
            world.add_plugin(plugin)

    return world


async def _teardown_world_async(world: World) -> None:
    if world and world.has_resource(DatabaseManager):
        db_mngr = world.get_resource(DatabaseManager)
        if db_mngr and db_mngr.engine:
            async with db_mngr.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            await db_mngr.engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_world_alpha(test_config: Config) -> AsyncGenerator[World, None]:
    """Create a test world named 'alpha'."""
    world = await _setup_world("test_world_alpha", test_config, plugins=None)
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
        super().__init__()  # type: ignore
        self.model_name = model_name_or_path
        self.dim = 384

    def encode(self, sentences: str | list[str], **_kwargs: Any) -> np.ndarray:
        """Encode sentences into embeddings."""
        is_single = isinstance(sentences, str)
        if is_single:
            sentences = [sentences]  # pyright: ignore[reportUnnecessaryComparison]

        embeddings: list[np.ndarray] = []
        for s in sentences:
            sum_ords = sum(ord(c) for c in s)
            vec = np.full(self.dim, float(sum_ords % 1000), dtype=np.float32)
            embeddings.append(vec)

        result = np.array(embeddings, dtype=np.float32)
        return result[0] if is_single else result


@pytest_asyncio.fixture(scope="function")
async def test_world_beta(test_config: Config) -> AsyncGenerator[World, None]:
    """Create a test world named 'beta'."""
    world = await _setup_world("test_world_beta", test_config)
    yield world
    await _teardown_world_async(world)


@pytest_asyncio.fixture(scope="function")
async def test_world_gamma(test_config: Config) -> AsyncGenerator[World, None]:
    """Create a test world named 'gamma'."""
    world = await _setup_world("test_world_gamma", test_config)
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


@pytest.fixture
def sample_image_a(tmp_path: Path) -> Path:
    """Create a sample image file."""
    file_path = tmp_path / "sample_A.png"
    img = Image.new("RGB", (2, 1), color=(128, 128, 128))
    img.save(file_path)
    return file_path


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a sample text file."""
    file_path = tmp_path / "sample_doc.txt"
    file_path.write_text("This is a common test document.")
    return file_path
