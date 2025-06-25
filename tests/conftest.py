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

# Store original settings
original_dam_worlds = global_settings.DAM_WORLDS
original_default_world_name = global_settings.DEFAULT_WORLD_NAME
original_testing_mode = global_settings.TESTING_MODE


@pytest.fixture(scope="session")
def test_worlds_config_data():
    """Provides the raw configuration dictionary for test worlds."""
    # Create temporary directories for asset storage for each test world
    # These will be cleaned up by the OS or manually if needed, but for fixtures,
    # it's often better to clean up in the fixture itself if state persists.
    # For session scope, these paths will exist for the whole session.
    # We'll use function-scoped temp dirs for asset paths to ensure test isolation for file operations.
    return {
        "test_world_alpha": {
            "DATABASE_URL": "sqlite:///:memory:?world=alpha",  # In-memory, unique per world
            # ASSET_STORAGE_PATH will be overridden per test function needing it
        },
        "test_world_beta": {
            "DATABASE_URL": "sqlite:///:memory:?world=beta",
        },
        "test_world_gamma": {  # Added for split tests
            "DATABASE_URL": "sqlite:///:memory:?world=gamma",
        },
        # Worlds for deletion tests to keep main test worlds clean if needed
        "test_world_alpha_del_split": {
            "DATABASE_URL": "sqlite:///:memory:?world=alpha_del_split",
        },
        "test_world_beta_del_split": {
            "DATABASE_URL": "sqlite:///:memory:?world=beta_del_split",
        },
        "test_world_gamma_del_split": {
            "DATABASE_URL": "sqlite:///:memory:?world=gamma_del_split",
        },
    }


@pytest.fixture(scope="function")
def settings_override(test_worlds_config_data, monkeypatch):
    """
    Overrides application settings for the duration of a test function.
    Each test world gets its own temporary asset storage path.
    """
    temp_storage_dirs = {}
    updated_test_worlds_config = {}

    for world_name, config in test_worlds_config_data.items():
        temp_dir = tempfile.mkdtemp(prefix=f"dam_test_{world_name}_")
        temp_storage_dirs[world_name] = Path(temp_dir)
        updated_test_worlds_config[world_name] = {
            **config,
            "ASSET_STORAGE_PATH": str(temp_dir),
        }

    # Use a unique default world for testing to avoid conflicts if some tests don't specify a world
    default_test_world = "test_world_alpha"

    # Override the global settings object by monkeypatching its attributes directly
    # This is generally safer than trying to reload a Pydantic settings object mid-flight
    # if modules have already imported `settings` from `dam.core.config`.

    # We need to ensure that the `settings` object itself is updated, or a new one
    # is created and used by the application during the test.
    # Pydantic settings are often instantiated once at import time.
    # The cleanest way is to control the environment variables Pydantic reads,
    # or to directly patch the `settings` instance.

    # Create a new Settings instance with overridden values
    # This ensures that the model_validator in Settings is run with the new values

    new_settings = Settings(
        DAM_WORLDS=json.dumps(updated_test_worlds_config),
        DAM_DEFAULT_WORLD_NAME=default_test_world,
        TESTING_MODE=True,
        # Ensure other critical settings are preserved or set to test defaults if necessary
    )

    # Monkeypatch the global `settings` instance in `dam.core.config`
    monkeypatch.setattr(dam.core.config, "settings", new_settings)

    # Also monkeypatch where db_manager might have already captured settings if it's module-scoped
    # This is tricky. It's better if db_manager is instantiated after settings are patched,
    # or if it can re-read settings. Assuming db_manager is function-scoped or re-initializable.

    yield new_settings  # Provide the overridden settings to the test

    # Restore original settings after test
    monkeypatch.setattr(dam.core.config, "settings", global_settings)  # Restore the original global instance

    # Clean up temporary asset storage directories
    for path in temp_storage_dirs.values():
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="function")
def test_db_manager(settings_override):
    """
    Provides a DatabaseManager instance configured with test worlds.
    Ensures tables are created and dropped for each world's engine.
    This fixture depends on `settings_override` to ensure settings are patched first.
    """
    # settings_override has already patched global `dam.core.config.settings`
    # So, DatabaseManager will pick up the patched settings when instantiated.
    # Pass the overridden settings object to the DatabaseManager constructor.
    manager = DatabaseManager(settings_override)

    # Create tables for all configured test worlds
    for world_name in manager.get_all_world_names():
        engine = manager.get_engine(world_name)
        Base.metadata.create_all(bind=engine)
        # print(f"Created tables for test world {world_name} on engine {engine.url}")

    yield manager

    # Drop tables for all configured test worlds after the test
    for world_name in manager.get_all_world_names():
        engine = manager.get_engine(world_name)
        Base.metadata.drop_all(bind=engine)
        # print(f"Dropped tables for test world {world_name} on engine {engine.url}")
        # Explicitly dispose of the engine to close in-memory DB connections if any issue
        engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_db_manager, settings_override):
    """
    Provides a SQLAlchemy session for the default test world ("test_world_alpha").
    This is a convenience fixture for tests that don't need to manage multiple worlds explicitly.
    The session is closed automatically after the test.
    """
    # settings_override ensures that global_settings.DEFAULT_WORLD_NAME is patched
    # to our desired default test world ("test_world_alpha")
    default_test_world_name = dam.core.config.settings.DEFAULT_WORLD_NAME
    if not default_test_world_name:  # Should be set by settings_override
        raise ValueError("Default test world name not set in overridden settings.")

    session = test_db_manager.get_db_session(default_test_world_name)
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def another_db_session(test_db_manager):
    """
    Provides a SQLAlchemy session for a secondary test world ("test_world_beta").
    Useful for testing interactions or isolation between two worlds.
    The session is closed automatically after the test.
    """
    session = test_db_manager.get_db_session("test_world_beta")
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
    file_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2avc1mp41") # Minimal MP4-like start
    return file_path

@pytest.fixture
def sample_audio_file_placeholder(tmp_path: Path) -> Path:
    """Creates a placeholder file with .mp3 extension for tests needing an audio file path."""
    file_path = tmp_path / "sample_audio_placeholder.mp3"
    file_path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x0f\x00") # Minimal MP3-like start
    return file_path

@pytest.fixture
def sample_gif_file_placeholder(tmp_path: Path) -> Path:
    """Creates a placeholder file with .gif extension. For actual GIF content, use Pillow."""
    # A very minimal valid GIF (1x1 transparent pixel)
    gif_bytes = bytes.fromhex(
        "47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b"
    )
    file_path = tmp_path / "sample_gif_placeholder.gif"
    file_path.write_bytes(gif_bytes)
    return file_path
