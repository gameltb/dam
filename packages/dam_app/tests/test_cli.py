import asyncio
import json
from functools import partial
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    Iterator,
    Optional,
)

import pytest
import pytest_asyncio
from dam.core.config import Settings
from dam.core.config import settings as global_settings
from dam.core.database import DatabaseManager
from dam.core.world import (
    World,
    clear_world_registry,
    create_and_register_world,
)
from dam.models.core.base_class import Base
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner, Result

from dam_app.cli import app

TEST_DEFAULT_WORLD_NAME = "cli_test_world_default"
TEST_ALPHA_WORLD_NAME = "cli_test_world_alpha"
TEST_BETA_WORLD_NAME = "cli_test_world_beta"

_original_settings_values = {}


@pytest.fixture(scope="session", autouse=True)
def backup_original_settings():
    _original_settings_values["DAM_WORLDS_CONFIG"] = global_settings.DAM_WORLDS_CONFIG
    _original_settings_values["worlds"] = global_settings.worlds.copy()
    _original_settings_values["DEFAULT_WORLD_NAME"] = global_settings.DEFAULT_WORLD_NAME
    _original_settings_values["TESTING_MODE"] = global_settings.TESTING_MODE
    yield


@pytest.fixture(scope="session")
def test_worlds_config_data_factory(tmp_path_factory):
    def _factory():
        return {
            "test_world_alpha": {
                "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path_factory.mktemp('alpha_db')}/test_alpha.db"
            },
            "test_world_beta": {
                "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path_factory.mktemp('beta_db')}/test_beta.db"
            },
            "test_world_gamma": {
                "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path_factory.mktemp('gamma_db')}/test_gamma.db"
            },
            "test_world_alpha_del_split": {
                "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path_factory.mktemp('alpha_del_split_db')}/test_alpha_del_split.db"
            },
            "test_world_beta_del_split": {
                "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path_factory.mktemp('beta_del_split_db')}/test_beta_del_split.db"
            },
            "test_world_gamma_del_split": {
                "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path_factory.mktemp('gamma_del_split_db')}/test_gamma_del_split.db"
            },
        }

    return _factory


@pytest.fixture(scope="function")
def settings_override(test_worlds_config_data_factory, monkeypatch, tmp_path) -> Generator[Settings, None, None]:
    temp_storage_dirs = {}
    raw_world_configs = test_worlds_config_data_factory()
    updated_test_worlds_config = {}

    for world_name, config_template in raw_world_configs.items():
        asset_temp_dir = tmp_path / f"assets_{world_name}"
        asset_temp_dir.mkdir(parents=True, exist_ok=True)
        temp_storage_dirs[world_name] = asset_temp_dir
        updated_test_worlds_config[world_name] = {
            **config_template,
            "ASSET_STORAGE_PATH": str(asset_temp_dir),
        }

    default_test_world_name = "test_world_alpha"
    new_settings = Settings(
        DAM_WORLDS_CONFIG=json.dumps(updated_test_worlds_config),
        DAM_DEFAULT_WORLD_NAME=default_test_world_name,
        TESTING_MODE=True,
    )

    original_settings_instance = global_settings
    monkeypatch.setattr("dam.core.config.settings", new_settings)
    clear_world_registry()
    yield new_settings
    monkeypatch.setattr("dam.core.config.settings", original_settings_instance)
    clear_world_registry()


async def _setup_world(world_name: str, settings_override_fixture: Settings) -> World:
    world = create_and_register_world(world_name, app_settings=settings_override_fixture)
    await world.create_db_and_tables()
    from dam.core.world_setup import register_core_systems

    register_core_systems(world)
    return world


async def _teardown_world_async(world: World):
    if world and world.has_resource(DatabaseManager):
        db_mngr = world.get_resource(DatabaseManager)
        if db_mngr and db_mngr.engine:
            async with db_mngr.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            await db_mngr.engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_world_alpha(settings_override: Settings) -> AsyncGenerator[World, None]:
    world = await _setup_world("test_world_alpha", settings_override)
    yield world
    await _teardown_world_async(world)


@pytest.mark.asyncio
async def test_cli_add_asset(test_world_alpha: World, temp_asset_file: Path):
    """Test the add-asset command."""
    from dam_app.cli import cli_add_asset, global_state
    from typer import Context

    # Set the global state for the world
    global_state.world_name = test_world_alpha.name

    # Create a mock context
    mock_ctx = Context(command=cli_add_asset)

    await cli_add_asset(
        ctx=mock_ctx,
        path_str=str(temp_asset_file),
        recursive=False,
    )

    # Verify that the asset was added
    from dam.functions import ecs_functions
    from dam_fs.models import FilePropertiesComponent

    async with test_world_alpha.db_session_maker() as session:
        entities = await ecs_functions.get_entities_with_component(session, FilePropertiesComponent)
        assert len(entities) == 1
        entity = entities[0]
        fp_component = await ecs_functions.get_component(session, entity.id, FilePropertiesComponent)
        assert fp_component is not None
        assert fp_component.original_filename == temp_asset_file.name


def test_cli_list_worlds(settings_override, capsys):
    """Test the list-worlds command."""
    from dam_app.cli import cli_list_worlds, create_and_register_all_worlds_from_settings

    # Ensure worlds are registered
    create_and_register_all_worlds_from_settings(settings_override)

    cli_list_worlds()

    captured = capsys.readouterr()
    assert "test_world_alpha" in captured.out
    assert "test_world_beta" in captured.out


import numpy as np


class MockSentenceTransformer:
    def __init__(self, model_name_or_path=None, **kwargs):
        self.model_name = model_name_or_path
        if model_name_or_path and "clip" in model_name_or_path.lower():
            self.dim = 512
        elif model_name_or_path and "MiniLM-L6-v2" in model_name_or_path:
            self.dim = 384
        else:
            self.dim = 384

    def encode(self, sentences, convert_to_numpy=True, **kwargs):
        original_sentences_type = type(sentences)
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = []
        for s in sentences:
            if not s or not s.strip():
                embeddings.append(np.zeros(self.dim, dtype=np.float32))
                continue
            sum_ords = sum(ord(c) for c in s)
            model_ord_sum = sum(ord(c) for c in (self.model_name or "default"))
            vec_elements = [sum_ords % 100, len(s) % 100, model_ord_sum % 100]
            if self.dim >= 3:
                vec = np.array(vec_elements[: self.dim] + [0.0] * (self.dim - min(3, self.dim)), dtype=np.float32)
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
        else:
            return np.array(embeddings) if convert_to_numpy else embeddings


@pytest.fixture(autouse=True, scope="function")
def global_mock_sentence_transformer_loader(monkeypatch):
    from dam_semantic import semantic_functions as semantic_service

    def mock_load_sync(model_name_str: str, model_load_params: Optional[Dict[str, Any]] = None):
        return MockSentenceTransformer(model_name_or_path=model_name_str, **(model_load_params or {}))

    monkeypatch.setattr(semantic_service, "_load_sentence_transformer_model_sync", mock_load_sync)


@pytest.fixture(scope="session", autouse=True)
def configure_session_logging():
    import logging

    original_levels = {}
    root_logger = logging.getLogger()
    original_levels["root"] = root_logger.level
    root_logger.setLevel(logging.WARNING)
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.WARNING)
    yield
    root_logger.setLevel(original_levels.get("root", logging.INFO))
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        original_level = original_levels.get(logger_name, logging.INFO)
        logger.setLevel(original_level)


@pytest_asyncio.fixture(scope="function")
async def test_world_beta(settings_override: Settings) -> AsyncGenerator[World, None]:
    world = await _setup_world("test_world_beta", settings_override)
    yield world
    await _teardown_world_async(world)


@pytest_asyncio.fixture(scope="function")
async def test_world_gamma(settings_override: Settings) -> AsyncGenerator[World, None]:
    world = await _setup_world("test_world_gamma", settings_override)
    yield world
    await _teardown_world_async(world)


@pytest_asyncio.fixture(scope="function")
async def db_session(test_world_alpha: World) -> AsyncGenerator[AsyncSession, None]:
    db_mngr = test_world_alpha.get_resource(DatabaseManager)
    async with db_mngr.session_local() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def another_db_session(test_world_beta: World) -> AsyncGenerator[AsyncSession, None]:
    db_mngr = test_world_beta.get_resource(DatabaseManager)
    async with db_mngr.session_local() as session:
        yield session


@pytest.fixture
def temp_asset_file(tmp_path):
    file_path = tmp_path / "test_asset.txt"
    file_path.write_text("This is a test asset.")
    return file_path


@pytest.fixture
def temp_image_file(tmp_path):
    from PIL import Image

    file_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (60, 30), color="red")
    img.save(file_path)
    return file_path


@pytest.fixture
def sample_image_a(tmp_path: Path) -> Path:
    img_a_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR42mNkgIL/DAwM/wUADgAB/vA/cQAAAABJRU5ErkJggg=="
    file_path = tmp_path / "sample_A.png"
    import base64

    file_path.write_bytes(base64.b64decode(img_a_b64))
    return file_path


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample_doc.txt"
    file_path.write_text("This is a common test document.")
    return file_path


@pytest.fixture
def sample_video_file_placeholder(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample_video_placeholder.mp4"
    file_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2avc1mp41")
    return file_path


@pytest.fixture
def sample_audio_file_placeholder(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample_audio_placeholder.mp3"
    file_path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x0f\x00")
    return file_path


@pytest.fixture
def sample_gif_file_placeholder(tmp_path: Path) -> Path:
    gif_bytes = bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b")
    file_path = tmp_path / "sample_gif_placeholder.gif"
    file_path.write_bytes(gif_bytes)
    return file_path


@pytest_asyncio.fixture(scope="function")
async def test_world_with_db_session(settings_override: Settings) -> AsyncGenerator[World, None]:
    world = await _setup_world("test_world_alpha", settings_override)
    yield world
    await _teardown_world_async(world)


