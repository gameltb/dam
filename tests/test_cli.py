import json
import os  # Added for os.path.exists
from pathlib import Path

import pytest
from typer.testing import CliRunner

# Import the app after patches, if any, are applied.
# For now, direct import is fine as we're starting simple.
from dam.cli import app
from dam.core.config import Settings, WorldConfig
from dam.core.config import settings as global_app_settings  # Added WorldConfig

# app_db_manager is no longer a global instance in dam.core.database
# Tests will use the db_manager instance created within the test_environment fixture.
from dam.core.database import DatabaseManager
from dam.core.world import (
    clear_world_registry,
)
from dam.models.base_class import Base as AppBase

runner = CliRunner()

TEST_DEFAULT_WORLD_NAME = "cli_test_world_default"
TEST_ALPHA_WORLD_NAME = "cli_test_world_alpha"
TEST_BETA_WORLD_NAME = "cli_test_world_beta"


@pytest.fixture(scope="function")
def test_environment(tmp_path: Path, monkeypatch):
    """
    Sets up a clean test environment for each test function.
    - Creates temporary directories for asset storage for multiple worlds.
    - Sets environment variables for DAM_WORLDS_CONFIG and DAM_DEFAULT_WORLD_NAME.
    - Patches app_settings and app_db_manager to use test-specific configurations.
    - Ensures databases are created for each test world.
    - Clears the world registry before and after each test.
    """
    clear_world_registry()  # Clear before test

    test_worlds_config = {}
    # Use global_app_settings for accessing original values
    original_dam_worlds_config = global_app_settings.DAM_WORLDS_CONFIG
    original_default_world_name = global_app_settings.DEFAULT_WORLD_NAME
    # original_dam_log_level is not part of Settings model, it's read from env by logging_config
    original_testing_mode = global_app_settings.TESTING_MODE

    # Create unique DB and storage for default, alpha, and beta test worlds
    for world_name_key in [TEST_DEFAULT_WORLD_NAME, TEST_ALPHA_WORLD_NAME, TEST_BETA_WORLD_NAME]:
        db_file = tmp_path / f"{world_name_key}.db"
        storage_dir = tmp_path / f"asset_storage_{world_name_key}"
        storage_dir.mkdir(parents=True, exist_ok=True)
        test_worlds_config[world_name_key] = {
            "DATABASE_URL": f"sqlite:///{db_file.resolve()}",
            "ASSET_STORAGE_PATH": str(storage_dir.resolve()),
        }

    # Monkeypatch environment variables that Settings will load
    monkeypatch.setenv("DAM_WORLDS_CONFIG", json.dumps(test_worlds_config))
    monkeypatch.setenv("DAM_DEFAULT_WORLD_NAME", TEST_DEFAULT_WORLD_NAME)
    monkeypatch.setenv("DAM_LOG_LEVEL", "DEBUG")  # Ensure consistent log level for tests
    monkeypatch.setenv("TESTING_MODE", "True")

    # Create new Settings and DatabaseManager instances based on the patched env vars
    # This simulates how the application would load settings at startup.
    new_settings = Settings()  # This will load from our monkeypatched env vars

    # Patch the global settings object that the application code (e.g., CLI, world creation) will import and use.
    # global_app_settings is the alias for 'from dam.core.config import settings'
    monkeypatch.setattr(global_app_settings, "DAM_WORLDS_CONFIG", new_settings.DAM_WORLDS_CONFIG)  # Patch the source
    monkeypatch.setattr(global_app_settings, "DEFAULT_WORLD_NAME", new_settings.DEFAULT_WORLD_NAME)
    monkeypatch.setattr(global_app_settings, "TESTING_MODE", new_settings.TESTING_MODE)
    # DAM_LOG_LEVEL is handled by env var directly by logging_config, not part of Settings model for patching here
    # Crucially, re-trigger the parsing of worlds in the global settings instance
    # Pydantic V2 might require re-initialization or careful modification if model_rebuild doesn't work as expected
    # For now, let's try patching the 'worlds' attribute directly after new_settings has parsed them.
    # This assumes that other parts of the app will use this patched 'worlds' dict.
    monkeypatch.setattr(global_app_settings, "worlds", new_settings.worlds)

    # new_db_manager is created here for tests to directly use if they need to manipulate DBs outside CLI commands,
    # or to get engines for table creation.
    # The CLI itself will cause Worlds to be created, and those Worlds will instantiate their own DatabaseManagers
    # using the (patched) global_app_settings.
    test_fixture_db_managers = {}
    for wn, wc in new_settings.worlds.items():
        test_fixture_db_managers[wn] = DatabaseManager(world_config=wc, testing_mode=new_settings.TESTING_MODE)

    # Ensure CLI's world initialization uses the patched global_app_settings.
    # The CLI's `main_callback` calls `create_and_register_all_worlds_from_settings(app_settings=global_app_settings)`.
    # This function should now use the patched global_app_settings.

    # Create all tables for all test worlds using the db_managers created for the fixture
    for world_name_setup, db_mgr_instance in test_fixture_db_managers.items():
        AppBase.metadata.create_all(bind=db_mgr_instance.engine)

    yield {
        "tmp_path": tmp_path,
        "settings": new_settings,  # provide the new settings object to tests
        "db_managers": test_fixture_db_managers,  # provide the map of db_managers
        "default_world_name": TEST_DEFAULT_WORLD_NAME,
        "alpha_world_name": TEST_ALPHA_WORLD_NAME,
        "beta_world_name": TEST_BETA_WORLD_NAME,
    }

    # Teardown: clear world registry and restore original settings if necessary
    clear_world_registry()
    # Restore global_app_settings to its original state
    monkeypatch.setattr(global_app_settings, "DAM_WORLDS_CONFIG", original_dam_worlds_config)
    monkeypatch.setattr(global_app_settings, "DEFAULT_WORLD_NAME", original_default_world_name)
    # DAM_LOG_LEVEL was not on Settings model, so no need to restore it here. Env var will be reset by monkeypatch.
    monkeypatch.setattr(global_app_settings, "TESTING_MODE", original_testing_mode)
    # Re-parse worlds based on original config
    # Manually reconstruct the 'worlds' dict for global_app_settings based on its restored attributes.
    # This avoids calling Settings() again, which was problematic in teardown.
    # global_app_settings.DAM_WORLDS_CONFIG and global_app_settings.DEFAULT_WORLD_NAME have been restored.

    restored_raw_configs = {}
    config_source_str = global_app_settings.DAM_WORLDS_CONFIG  # This is the original JSON string or path
    if os.path.exists(config_source_str):
        try:
            with open(config_source_str, "r") as f:
                restored_raw_configs = json.load(f)
        except (IOError, json.JSONDecodeError):  # Should not happen if original config was valid
            restored_raw_configs = json.loads(
                global_app_settings.model_fields["DAM_WORLDS_CONFIG"].default
            )  # Fallback to schema default
    else:
        try:
            restored_raw_configs = json.loads(config_source_str)
        except json.JSONDecodeError:
            restored_raw_configs = json.loads(
                global_app_settings.model_fields["DAM_WORLDS_CONFIG"].default
            )  # Fallback to schema default

    final_restored_worlds: dict[str, WorldConfig] = {}
    for name, config_data in restored_raw_configs.items():
        if isinstance(config_data, dict):  # Basic check
            config_data_with_name = {"name": name, **config_data}
            try:
                final_restored_worlds[name] = WorldConfig(**config_data_with_name)
            except Exception:  # If a specific world config is bad, skip it for restoration
                pass  # Or log an error

    monkeypatch.setattr(global_app_settings, "worlds", final_restored_worlds)
    # Also ensure DEFAULT_WORLD_NAME is consistent with the now restored 'worlds'
    if global_app_settings.DEFAULT_WORLD_NAME not in final_restored_worlds and final_restored_worlds:
        # If original default is no longer valid for the truly original worlds config, reset to Pydantic default or first
        pydantic_field_default = global_app_settings.model_fields["DEFAULT_WORLD_NAME"].default
        if pydantic_field_default in final_restored_worlds:
            monkeypatch.setattr(global_app_settings, "DEFAULT_WORLD_NAME", pydantic_field_default)
        else:
            monkeypatch.setattr(
                global_app_settings,
                "DEFAULT_WORLD_NAME",
                sorted(final_restored_worlds.keys())[0] if final_restored_worlds else None,
            )

    # No global db_manager to restore, as it's instance-based within Worlds.

    # Clean up temporary directories created by the test environment
    # This is usually handled by tmp_path fixture itself.
    # for world_name_key in [TEST_DEFAULT_WORLD_NAME, TEST_ALPHA_WORLD_NAME, TEST_BETA_WORLD_NAME]:
    #     storage_dir = tmp_path / f"asset_storage_{world_name_key}"
    #     if storage_dir.exists():
    #         shutil.rmtree(storage_dir)


def test_cli_help(test_environment):
    """Test the main help message for the CLI."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage: dam-cli [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "Digital Asset Management System CLI" in result.output
    assert "add-asset" in result.output
    assert "list-worlds" in result.output
    assert "setup-db" in result.output


def test_cli_list_worlds(test_environment):
    """Test the list-worlds command."""
    # The test_environment fixture already sets up worlds.
    # The CLI's main_callback should initialize them based on the patched settings.
    result = runner.invoke(app, ["list-worlds"])
    assert result.exit_code == 0
    assert "Available ECS worlds:" in result.output
    assert TEST_DEFAULT_WORLD_NAME in result.output
    assert "(default)" in result.output  # Default world should be marked
    assert TEST_ALPHA_WORLD_NAME in result.output
    assert TEST_BETA_WORLD_NAME in result.output


# Other tests removed as per user request to isolate the failing test and submit.
# Keeping test_cli_help (passes) and test_cli_list_worlds (fails) for now.
