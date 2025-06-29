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
import dam.models
from dam.core.config import Settings
from dam.core.config import settings as global_settings
from dam.core.database import DatabaseManager
from dam.core.world import World, clear_world_registry, create_and_register_world  # E402: Moved higher
from dam.models import Base

# Store original settings values to be restored
_original_settings_values = {}


@pytest.fixture(scope="session", autouse=True)
def backup_original_settings():
    """Backup original settings at the start of the session."""
    # This is a bit manual; ideally Pydantic settings could be snapshot/restored more cleanly.
    # We're backing up fields that settings_override is known to change.
    _original_settings_values["DAM_WORLDS_CONFIG"] = global_settings.DAM_WORLDS_CONFIG
    _original_settings_values["worlds"] = global_settings.worlds.copy()  # shallow copy
    _original_settings_values["DEFAULT_WORLD_NAME"] = global_settings.DEFAULT_WORLD_NAME
    _original_settings_values["TESTING_MODE"] = global_settings.TESTING_MODE
    yield  # No specific type yielded by backup_original_settings
    # Restoration will be handled by settings_override's finalizer using monkeypatch,
    # but this backup is a safety net or for understanding original state if needed.


@pytest.fixture(scope="session")
def test_worlds_config_data_factory(tmp_path_factory):
    """
    Provides a factory to generate raw configuration dictionary for test worlds,
    ensuring unique asset storage paths for each world using tmp_path_factory for session scope.
    """

    def _factory():
        # These paths are created once per session if this factory is used by a session-scoped fixture.
        # If used by function-scoped, they are unique per function.
        # For asset paths, function scope is generally better for isolation.
        # However, the main settings_override is function-scoped, which will handle temp dirs per test.
        # This session-scoped factory is more for providing the *structure* of config data.
        # The actual temp asset paths will be generated within settings_override (function-scoped).
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
    """
    Overrides application settings for the duration of a test function.
    Each test world gets its own temporary asset storage path using the function-scoped tmp_path.
    """
    temp_storage_dirs = {}
    raw_world_configs = test_worlds_config_data_factory()  # Get base config structure
    updated_test_worlds_config = {}

    for world_name, config_template in raw_world_configs.items():
        # Use function-scoped tmp_path for asset storage to ensure isolation between tests
        asset_temp_dir = tmp_path / f"assets_{world_name}"
        asset_temp_dir.mkdir(parents=True, exist_ok=True)
        temp_storage_dirs[world_name] = asset_temp_dir
        updated_test_worlds_config[world_name] = {
            **config_template,  # Contains DB URL from factory
            "ASSET_STORAGE_PATH": str(asset_temp_dir),
        }

    default_test_world_name = "test_world_alpha"

    # Create a new Settings instance with overridden values
    # This ensures that the model_validator in Settings is run with the new values
    # We pass the JSON string to DAM_WORLDS_CONFIG as the Settings model expects
    new_settings = Settings(
        DAM_WORLDS_CONFIG=json.dumps(updated_test_worlds_config),
        DAM_DEFAULT_WORLD_NAME=default_test_world_name,
        TESTING_MODE=True,
    )

    # Monkeypatch the global `settings` instance in `dam.core.config`
    original_settings_instance = dam.core.config.settings
    monkeypatch.setattr(dam.core.config, "settings", new_settings)

    # Clear the world registry before tests that use settings_override to ensure clean state
    clear_world_registry()

    yield new_settings

    # Restore original settings instance
    monkeypatch.setattr(dam.core.config, "settings", original_settings_instance)

    # Clean up temporary asset storage directories (tmp_path itself is function-scoped and auto-cleaned)
    # but explicit shutil.rmtree can be more robust if needed, though often not necessary with tmp_path.
    # for path in temp_storage_dirs.values():
    #     if path.exists(): # Check existence before trying to remove
    #         shutil.rmtree(path, ignore_errors=True)

    clear_world_registry()  # Clear registry after test too


async def _setup_world(world_name: str, settings_override_fixture: Settings) -> World:  # Made async
    """Helper function to get/create and setup a world for testing."""
    # settings_override_fixture is the specific Settings instance for tests.
    # Pass it directly to world creation functions.
    # The clear_world_registry() call is already present in settings_override fixture,
    # but can be called here too for safety if _setup_world is called multiple times
    # within a single settings_override scope (though typically it's 1-to-1).
    # clear_world_registry() # Ensure clean state before creating this specific world
    world = create_and_register_world(world_name, app_settings=settings_override_fixture)
    # create_and_register_world now calls initialize_world_resources internally.
    await world.create_db_and_tables()  # Await the async method

    # Core systems are registered after basic world and resource setup.
    from dam.core.world_setup import register_core_systems  # Updated import

    register_core_systems(world)

    # Note: The original asset_ingestion_systems might become obsolete or be removed.
    # If they still contain other relevant systems, those would need to be registered too.
    # For now, focusing on the ones moved from asset_service.

    return world


async def _teardown_world_async(world: World):
    """Helper function to teardown a test world's database and dispose of the engine."""
    if world and world.has_resource(DatabaseManager):
        db_mngr = world.get_resource(DatabaseManager)
        if db_mngr and db_mngr.engine:  # Engine is now guaranteed to be AsyncEngine
            async with db_mngr.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            await db_mngr.engine.dispose()  # Dispose async engine
    # Asset storage path (tmp_path subdirectory) will be cleaned by tmp_path fixture


@pytest_asyncio.fixture(scope="function")  # Use pytest_asyncio.fixture
async def test_world_alpha(settings_override: Settings) -> AsyncGenerator[World, None]:  # Made async
    """Provides the 'test_world_alpha' World instance, fully set up."""
    world = await _setup_world("test_world_alpha", settings_override)  # Await async setup
    yield world
    await _teardown_world_async(world)  # Await async teardown


@pytest.fixture(scope="session", autouse=True)
def configure_session_logging():
    """
    Set the log level to WARNING for all loggers for the entire test session.
    This is a session-scoped fixture that runs automatically.
    """
    import logging

    # Store original levels to restore them later
    original_levels = {}

    # Configure root logger
    root_logger = logging.getLogger()
    original_levels["root"] = root_logger.level
    root_logger.setLevel(logging.WARNING)

    # Configure all existing loggers
    # Iterate over a copy of the dictionary keys in case it's modified during iteration elsewhere
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.WARNING)

    yield

    # Restore original log levels
    root_logger.setLevel(original_levels.get("root", logging.INFO))
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        original_level = original_levels.get(logger_name, logging.INFO)
        logger.setLevel(original_level)


@pytest_asyncio.fixture(scope="function")  # Use pytest_asyncio.fixture
async def test_world_beta(settings_override: Settings) -> AsyncGenerator[World, None]:  # Made async
    """Provides the 'test_world_beta' World instance, fully set up."""
    world = await _setup_world("test_world_beta", settings_override)  # Await async setup
    yield world
    await _teardown_world_async(world)  # Await async teardown


@pytest_asyncio.fixture(scope="function")  # Use pytest_asyncio.fixture
async def test_world_gamma(settings_override: Settings) -> AsyncGenerator[World, None]:  # Made async
    """Provides the 'test_world_gamma' World instance, fully set up."""
    world = await _setup_world("test_world_gamma", settings_override)  # Await async setup
    yield world
    await _teardown_world_async(world)  # Await async teardown


@pytest_asyncio.fixture(scope="function")  # Use pytest_asyncio.fixture
async def db_session(  # Made async
    test_world_alpha: World,  # This fixture is now async
) -> AsyncGenerator[AsyncSession, None]:  # Yield AsyncSession
    """
    Provides an SQLAlchemy AsyncSession for the default test world ("test_world_alpha").
    The session is managed by an async context manager.
    """
    db_mngr = test_world_alpha.get_resource(DatabaseManager)
    # Assuming session_local from DatabaseManager is configured for AsyncSession
    # when an async engine is used (which it should be by now).
    async with db_mngr.session_local() as session:  # Use async with
        yield session
    # Session is automatically closed by async context manager


@pytest_asyncio.fixture(scope="function")  # Use pytest_asyncio.fixture
async def another_db_session(test_world_beta: World) -> AsyncGenerator[AsyncSession, None]:  # Made async
    """
    Provides an SQLAlchemy AsyncSession for a secondary test world ("test_world_beta").
    Useful for testing interactions or isolation between two worlds.
    The session is closed automatically after the test.
    """
    session = test_world_beta.get_db_session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def temp_asset_file(tmp_path):
    """Creates a temporary dummy file and returns its Path object."""
    file_path = tmp_path / "test_asset.txt"
    file_path.write_text("This is a test asset.")
    return file_path


@pytest.fixture
def temp_image_file(tmp_path):
    """Creates a temporary dummy PNG image file and returns its Path object."""
    from PIL import Image

    file_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (60, 30), color="red")
    img.save(file_path)
    return file_path


# Common file fixtures moved here from other test files


@pytest.fixture
def sample_image_a(tmp_path: Path) -> Path:
    """Creates a simple PNG image for testing."""
    # Using a simple base64 encoded PNG to avoid PIL dependency for this basic fixture if possible,
    # but tests using it for perceptual hashing will still need PIL/imagehash.
    # This is a 2x1 pixel red PNG.
    img_a_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR42mNkgIL/DAwM/wUADgAB/vA/cQAAAABJRU5ErkJggg=="
    file_path = tmp_path / "sample_A.png"
    import base64

    file_path.write_bytes(base64.b64decode(img_a_b64))
    return file_path


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Creates a simple text file for testing."""
    file_path = tmp_path / "sample_doc.txt"
    file_path.write_text("This is a common test document.")
    return file_path


@pytest.fixture
def sample_video_file_placeholder(tmp_path: Path) -> Path:
    """Creates a placeholder file with .mp4 extension for tests needing a video file path."""
    file_path = tmp_path / "sample_video_placeholder.mp4"
    file_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2avc1mp41")  # Minimal MP4-like start
    return file_path


@pytest.fixture
def sample_audio_file_placeholder(tmp_path: Path) -> Path:
    """Creates a placeholder file with .mp3 extension for tests needing an audio file path."""
    file_path = tmp_path / "sample_audio_placeholder.mp3"
    file_path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x0f\x00")  # Minimal MP3-like start
    return file_path


@pytest.fixture
def sample_gif_file_placeholder(tmp_path: Path) -> Path:
    """Creates a placeholder file with .gif extension. For actual GIF content, use Pillow."""
    # A very minimal valid GIF (1x1 transparent pixel)
    gif_bytes = bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b")
    file_path = tmp_path / "sample_gif_placeholder.gif"
    file_path.write_bytes(gif_bytes)
    return file_path


@pytest_asyncio.fixture(scope="function")  # Use pytest_asyncio.fixture
async def test_world_with_db_session(settings_override: Settings) -> AsyncGenerator[World, None]:  # Made async
    """
    Provides a fully initialized World instance using the 'test_world_alpha' configuration
    from settings_override. This world has its DB created and systems registered.
    The underlying database and asset storage are function-scoped via settings_override.
    """
    # Using "test_world_alpha" as the default world for these system tests
    world = await _setup_world("test_world_alpha", settings_override)  # Await async setup
    yield world
    await _teardown_world_async(world)  # Await async teardown
