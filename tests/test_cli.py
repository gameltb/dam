import json
import os  # Added for os.path.exists
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import Result
from sqlalchemy import select  # Ensure select is imported for tests
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
from dam.models import Base as AppBase
from dam.models.core.entity import Entity
from dam.models.core.file_location_component import FileLocationComponent
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.properties.file_properties_component import FilePropertiesComponent
from dam.models.source_info import source_types
from dam.models.source_info.original_source_info_component import OriginalSourceInfoComponent
from dam.services import file_operations
from dam.services.file_storage import get_file_path

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
    monkeypatch.setenv("DAM_LOG_LEVEL", "WARNING")  # Set log level to WARNING for tests
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


@pytest.fixture
def click_runner(capsys: pytest.CaptureFixture[str]) -> Iterator[CliRunner]:
    """
    Convenience fixture to return a click.CliRunner for cli testing
    """

    class MyCliRunner(CliRunner):
        """Override CliRunner to disable capsys"""

        def invoke(self, *args, **kwargs) -> Result:
            # Way to fix https://github.com/pallets/click/issues/824
            with capsys.disabled():
                result = super().invoke(*args, **kwargs)
            return result

    yield MyCliRunner()


def test_cli_help(test_environment, click_runner):
    """Test the main help message for the CLI."""
    result = click_runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_cli_list_worlds(test_environment, click_runner):
    """Test the list-worlds command."""
    # The test_environment fixture already sets up worlds.
    # The CLI's main_callback should initialize them based on the patched settings.
    result = click_runner.invoke(app, ["list-worlds"])
    assert result.exit_code == 0, f"CLI Error: {result.output}"
    output = result.output
    assert "Available ECS worlds:" in output
    assert f"{TEST_DEFAULT_WORLD_NAME} (default)" in output
    assert TEST_ALPHA_WORLD_NAME in output
    assert TEST_BETA_WORLD_NAME in output

    # Filter for lines that are part of the actual list-worlds output, ignoring logs
    actual_list_lines = []
    # Flags to identify sections of output, as logs can be interspersed.
    # Some commands print "Operating on world:..." before their main output.
    # list-worlds specific output starts with "Available ECS worlds:"
    in_list_worlds_output_section = False

    for line in output.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if stripped_line == "Available ECS worlds:":
            in_list_worlds_output_section = True
            actual_list_lines.append(stripped_line)
            continue

        if in_list_worlds_output_section:
            # Lines starting with "- " are world entries
            if stripped_line.startswith("- "):
                actual_list_lines.append(stripped_line)
            # Optional "Note:" or "Warning:" lines that can follow the list
            elif stripped_line.startswith("Note:") or stripped_line.startswith("Warning:"):
                actual_list_lines.append(stripped_line)
            # If we hit a line that's clearly not part of list-worlds (e.g. another log, or different command output)
            # we might stop, but logs can be anywhere. So, we just collect what matches.

    # Expected lines from the list-worlds command itself
    expected_lines_content = [
        "Available ECS worlds:",
        f"- {TEST_DEFAULT_WORLD_NAME} (default)",
        f"- {TEST_ALPHA_WORLD_NAME}",
        f"- {TEST_BETA_WORLD_NAME}",
    ]

    # Check if all expected primary lines are present in the filtered output
    for expected_line in expected_lines_content:
        assert any(expected_line in actual_line for actual_line in actual_list_lines), (
            f"Expected line '{expected_line}' not found in CLI output. Found: {actual_list_lines}"
        )

    # Count only the primary items: header and the three specific worlds
    # This avoids issues if optional "Note:" or "Warning:" lines appear
    count_primary_items = 0
    if any("Available ECS worlds:" in line for line in actual_list_lines):
        count_primary_items += 1
    if any(f"- {TEST_DEFAULT_WORLD_NAME} (default)" in line for line in actual_list_lines):
        count_primary_items += 1
    if any(f"- {TEST_ALPHA_WORLD_NAME}" in line and "(default)" not in line for line in actual_list_lines):
        count_primary_items += 1
    if any(f"- {TEST_BETA_WORLD_NAME}" in line and "(default)" not in line for line in actual_list_lines):
        count_primary_items += 1

    assert count_primary_items == 4, (
        f"Expected 4 primary list items (header + 3 worlds), found {count_primary_items} in filtered lines: {actual_list_lines}"
    )


def _create_dummy_file(filepath: Path, content: str = "dummy content") -> Path:
    filepath.write_text(content)
    return filepath


def _create_dummy_image(filepath: Path, size=(32, 32), color="red") -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)

    # Add distinct features based on the base color to help perceptual hashes
    # size is (width, height), e.g., (32, 32)
    w, h = size

    if color == "red":
        # Red image: draw a white circle in the middle
        radius = w // 4
        cx, cy = w // 2, h // 2
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(bbox, fill="white")

    elif color == "darkred":  # For the "similar" image
        # Dark red image: draw a light gray, slightly smaller circle
        radius = w // 5
        cx, cy = w // 2, h // 2
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(bbox, fill="lightgray")
        # Add a small feature to make it not identical to a plain circle if base colors were same
        draw.point((1, 1), fill="white")

    elif color == "blue":
        # Blue image: make it grayscale with lines to be very different
        img = Image.new("L", size, color="blue")  # Start with blue to get a certain gray base
        img = img.convert("RGB")  # Convert to RGB so it can be saved as PNG like others
        draw = ImageDraw.Draw(img)  # Re-get draw object for the new image
        for i in range(0, h, 4):
            draw.line([(0, i), (w, i)], fill="white", width=1)

    # Fallback for any other color, just a diagonal line
    else:  # Should not be hit by current tests using "red", "darkred", "blue"
        draw.line([(0, 0), (w, h)], fill="black", width=1)

    img.save(filepath, "PNG")
    return filepath


def test_cli_setup_db(test_environment, click_runner):
    """Test the setup-db command."""
    default_world_name = test_environment["default_world_name"]
    db_file = test_environment["tmp_path"] / f"{default_world_name}.db"

    # Remove the db file if it exists from the fixture setup to test creation by command
    if db_file.exists():
        db_file.unlink()

    result = click_runner.invoke(app, ["--world", default_world_name, "setup-db"])
    assert result.exit_code == 0, f"CLI Error: {result.output}"
    # assert f"Setting up database for world: '{default_world_name}'" in result.output # Avoid stdout checks
    # assert f"Database setup complete for world: '{default_world_name}'" in result.output # Avoid stdout checks
    assert db_file.exists(), "Database file was not created by setup-db command."

    # Verify that tables are created
    db_manager = test_environment["db_managers"][default_world_name]
    with db_manager.get_db_session() as session:
        # Check for a known table, e.g., 'entities'
        from sqlalchemy import text

        result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='entities'"))
        assert result.scalar_one_or_none() == "entities", "Entities table not found after setup-db"


def test_cli_add_asset_single_file(test_environment, caplog, click_runner):
    """Test adding a single asset file."""
    caplog.set_level("INFO")
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]
    dummy_content = "hello world for single asset"
    dummy_file = _create_dummy_file(tmp_path / "test_asset_single.txt", dummy_content)

    result = click_runner.invoke(app, ["--world", default_world_name, "add-asset", str(dummy_file)])
    assert result.exit_code == 0, f"CLI Error: {result.output}"
    # Reduced stdout checks
    # assert f"Processing file 1/1: {dummy_file.name}" in result.output
    # assert f"Dispatched AssetFileIngestionRequested for {dummy_file.name}" in result.output # This is a good check though
    # assert "Post-ingestion systems completed" in result.output
    # assert "Summary" in result.output # Avoid asserting stdout
    # assert "Total files processed: 1" in result.output # Avoid asserting stdout
    # assert "Errors encountered: 0" in result.output # Avoid asserting stdout

    # Instead of checking logs, verify side effects directly.
    # Check for file in CAS and database entries.

    # Imports moved to module level

    world_config = test_environment["settings"].get_world_config(default_world_name)

    content_hash = file_operations.calculate_sha256(dummy_file)
    # Use public get_file_path to verify CAS storage
    cas_file_path = get_file_path(content_hash, world_config)
    assert cas_file_path is not None, f"Asset file with hash {content_hash} not found in CAS via get_file_path"
    assert cas_file_path.exists(), f"Asset file not found in CAS at {cas_file_path}"
    assert cas_file_path.read_text() == dummy_content

    # Verify database entries
    db_manager = test_environment["db_managers"][default_world_name]
    expected_props = file_operations.get_file_properties(dummy_file)

    with db_manager.get_db_session() as session:
        # Find entity by SHA256 hash (hex string converted to bytes for query)
        content_hash_bytes = bytes.fromhex(content_hash)
        stmt_hash = select(ContentHashSHA256Component).where(
            ContentHashSHA256Component.hash_value == content_hash_bytes
        )
        hash_component = session.execute(stmt_hash).scalar_one_or_none()
        assert hash_component is not None, "ContentHashSHA256Component not found in DB"
        assert hash_component.hash_value == content_hash_bytes

        entity_id = hash_component.entity_id
        entity = session.get(Entity, entity_id)
        assert entity is not None, "Entity not found in DB"

        # Check OriginalSourceInfoComponent
        osi_comp = session.execute(
            select(OriginalSourceInfoComponent).where(OriginalSourceInfoComponent.entity_id == entity_id)
        ).scalar_one_or_none()
        assert osi_comp is not None, "OriginalSourceInfoComponent not found"
        assert osi_comp.source_type == source_types.SOURCE_TYPE_LOCAL_FILE
        assert not hasattr(osi_comp, "original_filename"), "OSI should not have original_filename"
        assert not hasattr(osi_comp, "original_path"), "OSI should not have original_path"

        # Check FilePropertiesComponent
        fpc_comp = session.execute(
            select(FilePropertiesComponent).where(FilePropertiesComponent.entity_id == entity_id)
        ).scalar_one_or_none()
        assert fpc_comp is not None, "FilePropertiesComponent not found"
        assert fpc_comp.original_filename == expected_props[0]  # expected_props[0] is original_filename
        assert fpc_comp.file_size_bytes == expected_props[1]  # expected_props[1] is size_bytes
        assert fpc_comp.mime_type == expected_props[2]  # expected_props[2] is mime_type

        # Check FileLocationComponent
        stmt_loc = select(FileLocationComponent).where(FileLocationComponent.entity_id == entity_id)
        location_component = session.execute(stmt_loc).scalar_one_or_none()
        assert location_component is not None, "FileLocationComponent not found"
        assert location_component.contextual_filename == dummy_file.name
        assert location_component.storage_type == "local_cas"
        assert location_component.content_identifier == content_hash


def test_cli_add_asset_directory_recursive(test_environment, caplog, click_runner):
    """Test adding assets from a directory recursively."""
    caplog.set_level("INFO")
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]

    asset_dir = tmp_path / "asset_source"
    asset_dir.mkdir()
    sub_dir = asset_dir / "subfolder"
    sub_dir.mkdir()

    content1 = "content for file1 recursive"
    content2 = "content for file2 recursive"
    file1 = _create_dummy_file(asset_dir / "file1_rec.txt", content1)
    file2 = _create_dummy_file(sub_dir / "file2_rec.txt", content2)

    result = click_runner.invoke(app, ["--world", default_world_name, "add-asset", str(asset_dir), "--recursive"])
    assert result.exit_code == 0, f"CLI Error: {result.output}"
    # Reduced stdout checks (avoiding brittle assertions on console output)

    # Check for files in CAS and database entries
    from sqlalchemy import select

    # Imports are already at the top of the file from previous change
    # from dam.models.core.entity import Entity
    # from dam.models.core.file_location_component import FileLocationComponent
    # from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
    # from dam.models.properties.file_properties_component import FilePropertiesComponent
    # from dam.models.source_info.original_source_info_component import OriginalSourceInfoComponent
    # from dam.models.source_info import source_types
    # from dam.services import file_operations
    # from dam.services.file_storage import get_file_path

    world_config = test_environment["settings"].get_world_config(default_world_name)
    db_manager = test_environment["db_managers"][default_world_name]

    files_to_check = [
        {"path": file1, "content": content1},
        {"path": file2, "content": content2},
    ]

    with db_manager.get_db_session() as session:
        for file_info in files_to_check:
            f_path = file_info["path"]
            f_content = file_info["content"]
            content_hash = file_operations.calculate_sha256(f_path)
            content_hash_bytes = bytes.fromhex(content_hash)
            expected_props = file_operations.get_file_properties(f_path)

            # Check CAS
            cas_file_path = get_file_path(content_hash, world_config)
            assert cas_file_path is not None, (
                f"Asset {f_path.name} (hash {content_hash}) not found in CAS via get_file_path"
            )
            assert cas_file_path.exists(), f"Asset {f_path.name} not found in CAS at {cas_file_path}"
            assert cas_file_path.read_text() == f_content, f"Content mismatch for {f_path.name}"

            # Check Database
            stmt_hash = select(ContentHashSHA256Component).where(
                ContentHashSHA256Component.hash_value == content_hash_bytes
            )
            hash_component = session.execute(stmt_hash).scalar_one_or_none()
            assert hash_component is not None, f"ContentHashSHA256Component for {f_path.name} not found"
            assert hash_component.hash_value == content_hash_bytes

            entity_id = hash_component.entity_id
            entity = session.get(Entity, entity_id)
            assert entity is not None, f"Entity for {f_path.name} not found"

            # Check OriginalSourceInfoComponent
            osi_comp = session.execute(
                select(OriginalSourceInfoComponent).where(OriginalSourceInfoComponent.entity_id == entity_id)
            ).scalar_one_or_none()  # Assuming one OSI per entity for this test case
            assert osi_comp is not None, f"OriginalSourceInfoComponent for {f_path.name} not found"
            assert osi_comp.source_type == source_types.SOURCE_TYPE_LOCAL_FILE
            assert not hasattr(osi_comp, "original_filename"), (
                f"OSI for {f_path.name} should not have original_filename"
            )
            assert not hasattr(osi_comp, "original_path"), f"OSI for {f_path.name} should not have original_path"

            # Check FilePropertiesComponent
            fpc_comp = session.execute(
                select(FilePropertiesComponent).where(FilePropertiesComponent.entity_id == entity_id)
            ).scalar_one_or_none()
            assert fpc_comp is not None, f"FilePropertiesComponent for {f_path.name} not found"
            assert fpc_comp.original_filename == expected_props[0]
            assert fpc_comp.file_size_bytes == expected_props[1]
            assert fpc_comp.mime_type == expected_props[2]

            # Check FileLocationComponent
            # We might have multiple FLCs if the same content was added via different original paths/names before,
            # but for this test, each file is unique content.
            stmt_loc = (
                select(FileLocationComponent)
                .where(FileLocationComponent.entity_id == entity_id)
                .where(FileLocationComponent.contextual_filename == f_path.name)
            )
            location_component = session.execute(stmt_loc).scalar_one_or_none()
            assert location_component is not None, f"FileLocationComponent for {f_path.name} not found"
            assert location_component.storage_type == "local_cas"
            assert location_component.content_identifier == content_hash


def test_cli_add_asset_no_copy(test_environment, caplog, click_runner):
    """Test adding an asset with --no-copy."""
    caplog.set_level("INFO")
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]
    dummy_file = _create_dummy_file(tmp_path / "ref_asset.txt", "reference content")

    result = click_runner.invoke(app, ["--world", default_world_name, "add-asset", str(dummy_file), "--no-copy"])
    assert result.exit_code == 0, f"CLI Error: {result.output}"

    # Verify database entries for '--no-copy'
    from sqlalchemy import select

    # Imports already at top
    # from dam.models.core.file_location_component import FileLocationComponent
    # from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
    # from dam.services import file_operations
    # from dam.services.file_storage import get_file_path
    # from dam.models.properties.file_properties_component import FilePropertiesComponent
    # from dam.models.source_info.original_source_info_component import OriginalSourceInfoComponent
    # from dam.models.source_info import source_types

    world_config = test_environment["settings"].get_world_config(default_world_name)
    db_manager = test_environment["db_managers"][default_world_name]
    content_hash = file_operations.calculate_sha256(dummy_file)
    content_hash_bytes = bytes.fromhex(content_hash)
    expected_props = file_operations.get_file_properties(dummy_file)

    # Check that file is NOT in CAS
    cas_file_path = get_file_path(content_hash, world_config)
    assert cas_file_path is None, (
        f"Asset file with hash {content_hash} found in CAS ({cas_file_path}) via get_file_path when --no-copy was used. It should not exist in CAS."
    )

    with db_manager.get_db_session() as session:
        stmt_hash = select(ContentHashSHA256Component).where(
            ContentHashSHA256Component.hash_value == content_hash_bytes
        )
        hash_component = session.execute(stmt_hash).scalar_one_or_none()
        assert hash_component is not None, "ContentHashSHA256Component not found"
        assert hash_component.hash_value == content_hash_bytes
        entity_id = hash_component.entity_id

        # Check OriginalSourceInfoComponent
        osi_comp = session.execute(
            select(OriginalSourceInfoComponent).where(OriginalSourceInfoComponent.entity_id == entity_id)
        ).scalar_one_or_none()
        assert osi_comp is not None, "OriginalSourceInfoComponent not found for no-copy asset"
        assert osi_comp.source_type == source_types.SOURCE_TYPE_REFERENCED_FILE
        assert not hasattr(osi_comp, "original_filename"), "OSI for no-copy should not have original_filename"
        assert not hasattr(osi_comp, "original_path"), "OSI for no-copy should not have original_path"

        # Check FilePropertiesComponent
        fpc_comp = session.execute(
            select(FilePropertiesComponent).where(FilePropertiesComponent.entity_id == entity_id)
        ).scalar_one_or_none()
        assert fpc_comp is not None, "FilePropertiesComponent not found for no-copy asset"
        assert fpc_comp.original_filename == expected_props[0]
        assert fpc_comp.file_size_bytes == expected_props[1]
        assert fpc_comp.mime_type == expected_props[2]

        # Check FileLocationComponent
        stmt_loc = select(FileLocationComponent).where(FileLocationComponent.entity_id == entity_id)
        location_component = session.execute(stmt_loc).scalar_one_or_none()
        assert location_component is not None, "FileLocationComponent not found for no-copy asset"
        assert location_component.contextual_filename == dummy_file.name
        assert location_component.storage_type == "local_reference"
        assert location_component.physical_path_or_key == str(dummy_file.resolve())
        assert location_component.content_identifier == content_hash


def test_cli_add_asset_duplicate(test_environment, caplog, click_runner):
    """Test adding a duplicate asset."""
    caplog.set_level("INFO")
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]
    dummy_file = _create_dummy_file(tmp_path / "dup_asset.txt", "duplicate content")

    # Add first time
    res1 = click_runner.invoke(app, ["--world", default_world_name, "add-asset", str(dummy_file)])
    assert res1.exit_code == 0, f"CLI Error adding first asset: {res1.output}"

    # Add second time (duplicate)
    res2 = click_runner.invoke(app, ["--world", default_world_name, "add-asset", str(dummy_file)])
    assert res2.exit_code == 0, f"CLI Error adding duplicate asset: {res2.output}"

    # Verify database entries for duplicate handling
    from sqlalchemy import select

    # Imports at top
    # from dam.models.core.file_location_component import FileLocationComponent
    # from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
    # from dam.services import file_operations
    # from dam.models.properties.file_properties_component import FilePropertiesComponent
    # from dam.models.source_info.original_source_info_component import OriginalSourceInfoComponent
    # from dam.models.source_info import source_types

    db_manager = test_environment["db_managers"][default_world_name]
    content_hash = file_operations.calculate_sha256(dummy_file)
    content_hash_bytes = bytes.fromhex(content_hash)

    with db_manager.get_db_session() as session:
        # Expect only one ContentHashSHA256Component for this content
        stmt_hashes = select(ContentHashSHA256Component).where(
            ContentHashSHA256Component.hash_value == content_hash_bytes
        )
        hash_components = session.execute(stmt_hashes).scalars().all()
        assert len(hash_components) == 1, "Duplicate ContentHashSHA256Component found or none found"
        assert hash_components[0].hash_value == content_hash_bytes

        entity_id = hash_components[0].entity_id
        expected_props = file_operations.get_file_properties(dummy_file)

        # Check OriginalSourceInfoComponent - should still be one primary one from the first add.
        # The second add should link to the existing entity.
        # Depending on how OSI is handled for duplicates (e.g., if it adds another OSI for the same source path)
        # this might need adjustment. For now, assume one distinct OSI by source_type and potentially other unique factors not filename/path.
        # With filename/path removed from OSI, if the second add re-adds an OSI, it would be identical.
        # Let's assume the system is smart enough to not add a fully identical OSI.
        osi_comps = (
            session.execute(
                select(OriginalSourceInfoComponent).where(OriginalSourceInfoComponent.entity_id == entity_id)
            )
            .scalars()
            .all()
        )
        assert len(osi_comps) == 1, f"Expected 1 OriginalSourceInfoComponent, found {len(osi_comps)}"
        assert osi_comps[0].source_type == source_types.SOURCE_TYPE_LOCAL_FILE
        assert not hasattr(osi_comps[0], "original_filename")
        assert not hasattr(osi_comps[0], "original_path")

        # Check FilePropertiesComponent - should be one, from the first add.
        fpc_comp = session.execute(
            select(FilePropertiesComponent).where(FilePropertiesComponent.entity_id == entity_id)
        ).scalar_one_or_none()
        assert fpc_comp is not None, "FilePropertiesComponent not found for duplicate asset"
        assert fpc_comp.original_filename == expected_props[0]
        assert fpc_comp.file_size_bytes == expected_props[1]
        assert fpc_comp.mime_type == expected_props[2]

        # Check FileLocationComponent - current system behavior is to add a new FLC if path is identical.
        # The test originally asserted len == 1, implying de-duplication of FLC by path.
        # Let's stick to that assumption for now as it's safer. If the system *does* create
        # multiple FLCs for the same path for the same entity, this test would need adjustment.
        # The `handle_asset_file_ingestion_request` *does* check if FLC exists for the same entity
        # and physical_storage_path_suffix. So it should be 1.
        stmt_locs = (
            select(FileLocationComponent)
            .where(FileLocationComponent.entity_id == entity_id)
            .where(FileLocationComponent.contextual_filename == dummy_file.name)
        )
        location_components = session.execute(stmt_locs).scalars().all()
        assert len(location_components) == 1, (
            f"Expected 1 FileLocationComponent for the same file path, found {len(location_components)}"
        )
        assert location_components[0].storage_type == "local_cas"

        # Verify that the CAS file still exists and content is correct
        world_config = test_environment["settings"].get_world_config(default_world_name)
        # from dam.services.file_storage import get_file_path # Import at top
        cas_file_path = get_file_path(content_hash, world_config)
        assert cas_file_path is not None, (
            f"Asset file with hash {content_hash} not found in CAS via get_file_path after duplicate add"
        )
        assert cas_file_path.exists(), "Asset file not found in CAS after duplicate add"
        assert cas_file_path.read_text() == "duplicate content"


def test_cli_find_file_by_hash(test_environment, caplog, click_runner):
    """Test finding a file by its hash."""
    caplog.set_level("INFO")
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]
    dummy_content = "content for hashing"
    dummy_file = _create_dummy_file(tmp_path / "hash_test.txt", dummy_content)

    # Add the asset first
    add_result = click_runner.invoke(app, ["--world", default_world_name, "add-asset", str(dummy_file)])
    assert add_result.exit_code == 0, f"CLI Error: {add_result.output}"

    from dam.services.file_operations import calculate_md5, calculate_sha256

    sha256_hash = calculate_sha256(dummy_file)
    md5_hash = calculate_md5(dummy_file)

    # Test find by SHA256
    caplog.clear()
    result_sha256 = click_runner.invoke(app, ["--world", default_world_name, "find-file-by-hash", sha256_hash])
    assert result_sha256.exit_code == 0, f"CLI Error finding by SHA256: {result_sha256.output}"
    # Assert that the output contains information about the found asset
    # This still requires some output checking, but it's more about content than exact log messages.
    # A better approach would be for the CLI command to return structured data (e.g., JSON)
    # if an option is provided, or to have a more predictable output format for parsing.
    # For now, we'll check for key identifiers in the output.
    assert sha256_hash in result_sha256.output
    assert dummy_file.name in result_sha256.output

    # Test find by MD5
    result_md5 = click_runner.invoke(
        app, ["--world", default_world_name, "find-file-by-hash", md5_hash, "--hash-type", "md5"]
    )
    assert result_md5.exit_code == 0, f"CLI Error finding by MD5: {result_md5.output}"
    assert md5_hash in result_md5.output
    assert dummy_file.name in result_md5.output

    # Test find by providing file
    result_file = click_runner.invoke(
        app,
        [
            "--world",
            default_world_name,
            "find-file-by-hash",
            "dummy_arg_for_runner",
            "--file",
            str(dummy_file),
        ],  # dummy_arg is required by typer if not a flag
    )
    assert result_file.exit_code == 0, f"CLI Error finding by file: {result_file.output}"
    # The command output should indicate it used the hash from the file
    assert f"Calculated sha256 hash: {sha256_hash}" in result_file.output  # This is a reasonable output to check
    assert sha256_hash in result_file.output  # The found asset info
    assert dummy_file.name in result_file.output


def test_cli_find_similar_images(test_environment, caplog, click_runner):
    """Test finding similar images."""
    caplog.set_level("INFO")
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]

    img_dir = tmp_path / "image_assets"
    img_dir.mkdir(exist_ok=True)

    img1_path = _create_dummy_image(img_dir / "img1.png", color="red")
    img2_path = _create_dummy_image(img_dir / "img2.png", color="darkred")  # Similar
    img3_path = _create_dummy_image(img_dir / "img3.png", color="blue")  # Different

    # Add images
    for img_path in [img1_path, img2_path, img3_path]:
        add_res = click_runner.invoke(app, ["--world", default_world_name, "add-asset", str(img_path)])
        assert add_res.exit_code == 0, f"CLI Error adding image {img_path.name}: {add_res.output}"

    # Test find similar
    # Use a high threshold to ensure the slightly different red image is found
    result_similar = click_runner.invoke(
        app,
        [
            "--world",
            default_world_name,
            "find-similar-images",
            str(img1_path),
            "--phash-threshold",
            "10",
            "--ahash-threshold",
            "10",
            "--dhash-threshold",
            "10",  # Increased threshold to catch darkred vs red
        ],
    )
    assert result_similar.exit_code == 0, f"CLI Error finding similar images: {result_similar.output}"

    # Verify output contains the similar image (img2) and not the different one (img3)
    # This relies on the output format of the command.
    # A more robust test would involve checking database state or specific event outputs if available.
    output = result_similar.output
    assert img1_path.name in output, "Query image itself should be mentioned in output"
    assert img2_path.name in output, "Similar image (img2.png) not found in output"
    assert img3_path.name not in output, "Dissimilar image (img3.png) should not be in output"

    # Example of how pHash distances could be checked if the output included them:
    # For instance, if output was "Found similar image: img2.png (pHash dist: X, aHash dist: Y, dHash dist: Z)"
    # import re
    # match_img2 = re.search(rf"{img2_path.name} \(pHash dist: (\d+)", output)
    # assert match_img2 is not None, "pHash distance for img2 not found in output"
    # phash_dist_img2 = int(match_img2.group(1))
    # assert phash_dist_img2 <= 10 # Check against the threshold used

    # Test with default thresholds (likely stricter) - img2 might not be found
    result_strict = click_runner.invoke(
        app,
        [
            "--world",
            default_world_name,
            "find-similar-images",
            str(img1_path),
            # Using default thresholds
        ],
    )
    assert result_strict.exit_code == 0, f"CLI Error with strict thresholds: {result_strict.output}"
    # Depending on default thresholds and image similarity, img2 might or might not appear.
    # This part of the test might need adjustment based on actual imagehash behavior for the dummy images.
    # For now, let's assume default thresholds are strict enough that img2 (darkred vs red) might be excluded
    # or included with a small distance. The key is that the command runs.
    # If img2 IS found with default thresholds, then this is fine.
    # If it's NOT found, that's also fine if the default thresholds are indeed strict.
    # The main goal is that the command executes and provides some output.
    # A more precise test would require knowing the exact perceptual hash values of the generated dummy images.
    # For now, ensuring the command runs and the query image is in output is a basic check.
    assert img1_path.name in result_strict.output
