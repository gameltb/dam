"""Fixtures for DAM tests."""

import json
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
from dam.core.config import Settings
from dam.core.config import settings as global_settings
from dam.core.database import DatabaseManager
from dam.core.transaction import WorldTransaction
from dam.core.transaction_manager import TransactionManager
from dam.core.world import (
    World,
    clear_world_registry,
    create_and_register_world,
)
from dam.models.core.base_class import Base
from PIL import Image
from psycopg import sql
from scipy.io.wavfile import write as write_wav  # type: ignore[import]
from sqlalchemy.ext.asyncio import AsyncSession

_original_settings_values: dict[str, Any] = {}
MOCK_TRANSFORMER_EMBEDDING_DIM = 3


@pytest.fixture(scope="session", autouse=True)
def backup_original_settings() -> None:
    """Backup the original settings before tests run."""
    _original_settings_values["DAM_WORLDS_CONFIG"] = global_settings.DAM_WORLDS_CONFIG
    _original_settings_values["worlds"] = global_settings.worlds.copy()
    _original_settings_values["DEFAULT_WORLD_NAME"] = global_settings.DEFAULT_WORLD_NAME


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
        # Need to terminate all connections before dropping the database
        await conn.execute(
            sql.SQL("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s"), (db_name,)
        )
        await conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name)))
    except psycopg.errors.InvalidCatalogName:
        pass  # Database does not exist
    finally:
        await conn.close()


@pytest_asyncio.fixture(scope="function")
async def settings_override(test_db: str, monkeypatch: Any, tmp_path: Path) -> AsyncGenerator[Settings, None]:
    """Override settings for testing."""

    def _factory() -> dict[str, dict[str, str]]:
        return {
            "test_world_alpha": {"DATABASE_URL": test_db},
            "test_world_beta": {"DATABASE_URL": test_db},
            "test_world_gamma": {"DATABASE_URL": test_db},
            "test_world_alpha_del_split": {"DATABASE_URL": test_db},
            "test_world_beta_del_split": {"DATABASE_URL": test_db},
            "test_world_gamma_del_split": {"DATABASE_URL": test_db},
        }

    raw_world_configs = _factory()
    updated_test_worlds_config: dict[str, Any] = {}

    for world_name, config_template in raw_world_configs.items():
        asset_temp_dir = tmp_path / f"assets_{world_name}"
        asset_temp_dir.mkdir(parents=True, exist_ok=True)
        updated_test_worlds_config[world_name] = {
            **config_template,
            "ASSET_STORAGE_PATH": str(asset_temp_dir),
        }

    default_test_world_name = "test_world_alpha"
    new_settings = Settings(
        DAM_WORLDS_CONFIG=json.dumps(updated_test_worlds_config),
        DEFAULT_WORLD_NAME=default_test_world_name,
    )

    monkeypatch.setattr(global_settings, "DAM_WORLDS_CONFIG", new_settings.DAM_WORLDS_CONFIG)
    monkeypatch.setattr(global_settings, "worlds", new_settings.worlds)
    monkeypatch.setattr(global_settings, "DEFAULT_WORLD_NAME", new_settings.DEFAULT_WORLD_NAME)

    clear_world_registry()
    yield new_settings
    monkeypatch.setattr(global_settings, "DAM_WORLDS_CONFIG", _original_settings_values["DAM_WORLDS_CONFIG"])
    monkeypatch.setattr(global_settings, "worlds", _original_settings_values["worlds"])
    monkeypatch.setattr(global_settings, "DEFAULT_WORLD_NAME", _original_settings_values["DEFAULT_WORLD_NAME"])
    clear_world_registry()


async def _setup_world(world_name: str, settings_override_fixture: Settings, plugins: list[Any] | None = None) -> World:
    """Set up a dam world, with optional plugins."""
    world = create_and_register_world(world_name, app_settings=settings_override_fixture)
    world.add_resource(world, World)

    transaction_manager = world.get_context(WorldTransaction)
    assert isinstance(transaction_manager, TransactionManager)
    await transaction_manager.create_db_and_tables()

    if plugins:
        for plugin in plugins:
            world.add_plugin(plugin)

    return world


async def _teardown_world_async(world: World) -> None:
    if world and world.has_context(WorldTransaction):
        transaction_manager = world.get_context(WorldTransaction)
        assert isinstance(transaction_manager, TransactionManager)
        db_mngr = transaction_manager.db_manager
        if db_mngr and db_mngr.engine:
            async with db_mngr.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            await db_mngr.engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_world_alpha(settings_override: Settings) -> AsyncGenerator[World, None]:
    """Create a test world named 'alpha'."""
    world = await _setup_world("test_world_alpha", settings_override, plugins=None)
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
        if model_name_or_path and "clip" in model_name_or_path.lower():
            self.dim = 512
        elif model_name_or_path and "MiniLM-L6-v2" in model_name_or_path:
            self.dim = 384
        else:
            self.dim = 384

    def forward(self, features: Any) -> Any:
        """Perform a forward pass."""
        return features

    def encode(
        self, sentences: str | list[str], convert_to_numpy: bool = True, **_kwargs: Any
    ) -> np.ndarray | list[list[float]]:
        """Encode sentences into embeddings."""
        original_sentences_type = type(sentences)
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings: list[Any] = []
        for s in sentences:
            if not s or not s.strip():
                embeddings.append(np.zeros(self.dim, dtype=np.float32))
                continue
            sum_ords = sum(ord(c) for c in s)
            model_ord_sum = sum(ord(c) for c in (self.model_name or "default"))
            vec_elements = [sum_ords % 100, len(s) % 100, model_ord_sum % 100]
            if self.dim >= MOCK_TRANSFORMER_EMBEDDING_DIM:
                vec = np.array(
                    vec_elements[: self.dim] + [0.0] * (self.dim - min(MOCK_TRANSFORMER_EMBEDDING_DIM, self.dim)),
                    dtype=np.float32,
                )
            elif self.dim > 0:
                vec = np.array(vec_elements[: self.dim], dtype=np.float32)
            else:
                vec = np.array([], dtype=np.float32)
            if vec.shape[0] != self.dim and self.dim > 0:
                padding = np.zeros(self.dim - vec.shape[0], dtype=np.float32)
                vec = np.concatenate((vec, padding))
            elif vec.shape[0] != self.dim and self.dim == 0:
                vec = np.array([], dtype=np.float32)
            embeddings.append(vec)
        if not convert_to_numpy:
            embeddings = [e.tolist() for e in embeddings]
        if original_sentences_type is str:
            return embeddings[0] if embeddings else np.array([])
        return np.array(embeddings) if convert_to_numpy else embeddings


@pytest_asyncio.fixture(scope="function")
async def test_world_beta(settings_override: Settings) -> AsyncGenerator[World, None]:
    """Create a test world named 'beta'."""
    world = await _setup_world("test_world_beta", settings_override)
    yield world
    await _teardown_world_async(world)


@pytest_asyncio.fixture(scope="function")
async def test_world_gamma(settings_override: Settings) -> AsyncGenerator[World, None]:
    """Create a test world named 'gamma'."""
    world = await _setup_world("test_world_gamma", settings_override)
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


@pytest.fixture
def sample_video_file_placeholder(tmp_path: Path) -> Path:
    """Create a placeholder video file."""
    file_path = tmp_path / "sample_video_placeholder.mp4"
    # A minimal mp4 file
    file_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2avc1mp41")
    return file_path


@pytest.fixture
def sample_audio_file_placeholder(tmp_path: Path) -> Path:
    """Create a placeholder audio file."""
    file_path = tmp_path / "sample_audio_placeholder.wav"
    samplerate = 44100
    duration = 1
    frequency = 440
    t = np.linspace(0.0, duration, int(samplerate * duration))
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2.0 * np.pi * frequency * t)
    write_wav(file_path, samplerate, data.astype(np.int16))
    return file_path


@pytest.fixture
def sample_gif_file_placeholder(tmp_path: Path) -> Path:
    """Create a placeholder GIF file."""
    file_path = tmp_path / "sample_gif_placeholder.gif"
    img = Image.new("RGB", (1, 1), color=(255, 255, 255))
    img.save(file_path)
    return file_path


@pytest.fixture
def sample_wav_file(tmp_path: Path) -> Path:
    """Create a sample WAV file."""
    file_path = tmp_path / "sample.wav"
    samplerate = 48000
    duration = 1
    frequency = 440
    t = np.linspace(0.0, duration, int(samplerate * duration))
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2.0 * np.pi * frequency * t)
    write_wav(file_path, samplerate, data.astype(np.int16))
    return file_path
