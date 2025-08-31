import json
from pathlib import Path
from typing import (
    AsyncGenerator,  # Added for async generator type hint
    Generator,  # Added for fixture type hints
)

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession  # Added for AsyncSession type hint

# Ensure models are imported so Base knows about them for table creation
# This will also trigger component registration
import dam.models  # This line can sometimes be problematic if dam.models itself has top-level import issues
from dam.core.config import Settings
from dam.core.config import settings as global_settings
from dam.core.database import DatabaseManager
from dam.core.world import World, clear_world_registry, create_and_register_world
from dam.models.core.base_class import Base

# Store original settings values to be restored
_original_settings_values = {}


@pytest.fixture(scope="session", autouse=True)
def backup_original_settings():
    _original_settings_values["DAM_WORLDS_CONFIG"] = global_settings.DAM_WORLDS_CONFIG
    _original_settings_values["worlds"] = global_settings.worlds.copy()
    _original_settings_values["DEFAULT_WORLD_NAME"] = global_settings.DEFAULT_WORLD_NAME
    _original_settings_values["TESTING_MODE"] = global_settings.TESTING_MODE
    yield


from pytest_postgresql import factories

db_factory = factories.postgresql_noproc()
transacted_postgresql_db = factories.postgresql("db_factory")


@pytest.fixture(scope="function")
def settings_override(transacted_postgresql_db, monkeypatch, tmp_path) -> Generator[Settings, None, None]:
    info = transacted_postgresql_db.info
    db_url = f"postgresql+psycopg://{info.user}:{info.password}@{info.host}:{info.port}/{info.dbname}"

    world_configs = {
        "test_world_alpha": {
            "DATABASE_URL": db_url,
            "ASSET_STORAGE_PATH": str(tmp_path / "assets_alpha"),
        },
        "test_world_beta": {
            "DATABASE_URL": db_url,
            "ASSET_STORAGE_PATH": str(tmp_path / "assets_beta"),
        },
    }

    # Ensure asset storage paths exist
    (tmp_path / "assets_alpha").mkdir()
    (tmp_path / "assets_beta").mkdir()

    new_settings = Settings(
        DAM_WORLDS_CONFIG=json.dumps(world_configs),
        DAM_DEFAULT_WORLD_NAME="test_world_alpha",
        TESTING_MODE=True,
    )

    original_settings_instance = dam.core.config.settings
    monkeypatch.setattr(dam.core.config, "settings", new_settings)
    clear_world_registry()
    yield new_settings
    monkeypatch.setattr(dam.core.config, "settings", original_settings_instance)
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
