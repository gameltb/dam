import json
import shutil
import tempfile
from pathlib import Path

import pytest

# Ensure models are imported so Base knows about them for table creation
# This will also trigger component registration
import dam.models
from dam.core.config import Settings
from dam.core.config import settings as global_settings
from dam.core.database import DatabaseManager
from dam.models.base_class import Base

# Store original settings values to be restored
_original_settings_values = {}

# Import World for type hinting
from dam.core.world import World, create_and_register_world, clear_world_registry, get_world


@pytest.fixture(scope="session", autouse=True)
def backup_original_settings():
    """Backup original settings at the start of the session."""
    # This is a bit manual; ideally Pydantic settings could be snapshot/restored more cleanly.
    # We're backing up fields that settings_override is known to change.
    _original_settings_values["DAM_WORLDS_CONFIG_SOURCE"] = global_settings.DAM_WORLDS_CONFIG_SOURCE
    _original_settings_values["worlds"] = global_settings.worlds.copy() # shallow copy
    _original_settings_values["DEFAULT_WORLD_NAME"] = global_settings.DEFAULT_WORLD_NAME
    _original_settings_values["TESTING_MODE"] = global_settings.TESTING_MODE
    yield
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
            "test_world_alpha": {"DATABASE_URL": f"sqlite:///{tmp_path_factory.mktemp('alpha_db')}/test_alpha.db"},
            "test_world_beta": {"DATABASE_URL": f"sqlite:///{tmp_path_factory.mktemp('beta_db')}/test_beta.db"},
            "test_world_gamma": {"DATABASE_URL": f"sqlite:///{tmp_path_factory.mktemp('gamma_db')}/test_gamma.db"},
            "test_world_alpha_del_split": {"DATABASE_URL": f"sqlite:///{tmp_path_factory.mktemp('alpha_del_split_db')}/test_alpha_del_split.db"},
            "test_world_beta_del_split": {"DATABASE_URL": f"sqlite:///{tmp_path_factory.mktemp('beta_del_split_db')}/test_beta_del_split.db"},
            "test_world_gamma_del_split": {"DATABASE_URL": f"sqlite:///{tmp_path_factory.mktemp('gamma_del_split_db')}/test_gamma_del_split.db"},
        }
    return _factory


@pytest.fixture(scope="function")
def settings_override(test_worlds_config_data_factory, monkeypatch, tmp_path):
    """
    Overrides application settings for the duration of a test function.
    Each test world gets its own temporary asset storage path using the function-scoped tmp_path.
    """
    temp_storage_dirs = {}
    raw_world_configs = test_worlds_config_data_factory() # Get base config structure
    updated_test_worlds_config = {}

    for world_name, config_template in raw_world_configs.items():
        # Use function-scoped tmp_path for asset storage to ensure isolation between tests
        asset_temp_dir = tmp_path / f"assets_{world_name}"
        asset_temp_dir.mkdir(parents=True, exist_ok=True)
        temp_storage_dirs[world_name] = asset_temp_dir
        updated_test_worlds_config[world_name] = {
            **config_template, # Contains DB URL from factory
            "ASSET_STORAGE_PATH": str(asset_temp_dir),
        }

    default_test_world_name = "test_world_alpha"

    # Create a new Settings instance with overridden values
    # This ensures that the model_validator in Settings is run with the new values
    # We pass the JSON string to DAM_WORLDS_CONFIG_SOURCE as the Settings model expects
    new_settings = Settings(
        DAM_WORLDS_CONFIG_SOURCE=json.dumps(updated_test_worlds_config),
        DAM_DEFAULT_WORLD_NAME=default_test_world_name,
        TESTING_MODE=True,
    )

    # Monkeypatch the global `settings` instance in `dam.core.config`
    original_settings_instance = dam.core.config.settings
    monkeypatch.setattr(dam.core.config, "settings", new_settings)

    # Clear the world registry before tests that use settings_override to ensure clean state
    clear_world_registry()

    yield new_settings  # Provide the overridden settings to the test function

    # Restore original settings instance
    monkeypatch.setattr(dam.core.config, "settings", original_settings_instance)

    # Clean up temporary asset storage directories (tmp_path itself is function-scoped and auto-cleaned)
    # but explicit shutil.rmtree can be more robust if needed, though often not necessary with tmp_path.
    # for path in temp_storage_dirs.values():
    #     if path.exists(): # Check existence before trying to remove
    #         shutil.rmtree(path, ignore_errors=True)

    clear_world_registry() # Clear registry after test too


def _setup_world(world_name: str, settings_override_fixture: Settings) -> World:
    """Helper function to get/create and setup a world for testing."""
    # settings_override_fixture has already patched global settings
    # create_and_register_world will use these patched settings
    world = create_and_register_world(world_name)
    world.create_db_and_tables() # Ensure tables are created for this world's DB
    return world

def _teardown_world(world: World):
    """Helper function to teardown a test world."""
    if world and world.db_manager and world.db_manager.engine:
        Base.metadata.drop_all(bind=world.db_manager.engine)
        world.db_manager.engine.dispose() # Close connections
    # Asset storage path (tmp_path subdirectory) will be cleaned by tmp_path fixture

@pytest.fixture(scope="function")
def test_world_alpha(settings_override: Settings) -> World:
    """Provides the 'test_world_alpha' World instance, fully set up."""
    world = _setup_world("test_world_alpha", settings_override)
    yield world
    _teardown_world(world)

@pytest.fixture(scope="function")
def test_world_beta(settings_override: Settings) -> World:
    """Provides the 'test_world_beta' World instance, fully set up."""
    world = _setup_world("test_world_beta", settings_override)
    yield world
    _teardown_world(world)

@pytest.fixture(scope="function")
def test_world_gamma(settings_override: Settings) -> World:
    """Provides the 'test_world_gamma' World instance, fully set up."""
    world = _setup_world("test_world_gamma", settings_override)
    yield world
    _teardown_world(world)


@pytest.fixture(scope="function")
def db_session(test_world_alpha: World):
    """
    Provides a SQLAlchemy session for the default test world ("test_world_alpha").
    The session is closed automatically after the test.
    """
    session = test_world_alpha.get_db_session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def another_db_session(test_world_beta: World):
    """
    Provides a SQLAlchemy session for a secondary test world ("test_world_beta").
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
