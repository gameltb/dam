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
from dam.core.database import DatabaseManager
from dam.core.world import World
from dam.core.world_manager import create_world_from_components
from dam.models.config import ConfigComponent
from dam.plugins.core import CoreSettingsComponent
from PIL import Image
from psycopg import sql

from dam_test_utils.teardown import teardown_world_async
from dam_test_utils.types import WorldFactory

MOCK_TRANSFORMER_EMBEDDING_DIM = 3


@pytest.fixture(scope="session")
def test_db_factory() -> Any:
    """Factory for creating temporary databases for testing."""

    async def _test_db_factory(db_name_prefix: str) -> AsyncGenerator[str, None]:
        db_user = os.environ.get("POSTGRES_USER", "postgres")
        db_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
        db_host = os.environ.get("POSTGRES_HOST", "localhost")
        db_port = os.environ.get("POSTGRES_PORT", "5432")
        db_name = f"{db_name_prefix}_{uuid.uuid4().hex}"

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
            # Terminate all connections to the database to allow it to be dropped.
            await conn.execute(
                sql.SQL("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s"), (db_name,)
            )
            await conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name)))
        except psycopg.errors.InvalidCatalogName:
            pass  # Database already dropped
        finally:
            await conn.close()

    return _test_db_factory


@pytest_asyncio.fixture
async def world_factory(test_db_factory: Any, tmp_path_factory: Any) -> AsyncGenerator[WorldFactory, None]:
    """Pytest fixture that provides a factory for creating isolated World instances."""
    created_worlds: list[str] = []

    async def _create_world(world_name: str, components: list[ConfigComponent]) -> World:
        """Create a world with the given name and components."""
        # Ensure a core settings component with a unique DB is present.
        if not any(isinstance(c, CoreSettingsComponent) for c in components):
            temp_db_url_generator = test_db_factory(f"db_{world_name}")
            temp_db_url = await temp_db_url_generator.__anext__()
            temp_alembic_path = tmp_path_factory.mktemp(f"alembic_{world_name}")
            components.append(
                CoreSettingsComponent(
                    plugin_name="core",
                    database_url=temp_db_url,
                    alembic_path=str(temp_alembic_path),
                )
            )

        # Create the world using the core factory function
        world = create_world_from_components(world_name, components)
        await world.get_resource(DatabaseManager).create_db_and_tables()

        created_worlds.append(world.name)
        return world

    yield _create_world

    # Teardown: Unregister all worlds created by this factory
    for name in created_worlds:
        world = world_manager.get_world(name)
        if world:
            await teardown_world_async(world)
        world_manager.unregister_world(name)


class MockSentenceTransformer(torch.nn.Module):
    """A mock sentence transformer for testing."""

    def __init__(self, model_name_or_path: str | None = None, **_kwargs: Any) -> None:
        """Initialize the mock sentence transformer."""
        super().__init__()  # type: ignore
        self.model_name = model_name_or_path
        self.dim = 384

    def encode(self, sentences: str | list[str], **_kwargs: Any) -> np.ndarray:
        """Encode sentences into embeddings."""
        if isinstance(sentences, str):
            return np.zeros(self.dim, dtype=np.float32)
        return np.zeros((len(sentences), self.dim), dtype=np.float32)


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
