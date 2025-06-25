import hashlib
import os
from pathlib import Path

import pytest

from dam.core.config import settings
from dam.services import file_storage

# No longer using manage_test_storage_directory, settings_override from conftest.py handles this.
# Tests will need to use settings_override and specify world_name for file_storage calls.

# Using app_settings from dam.core.config which is patched by settings_override
from dam.core.config import settings as app_settings

def test_get_storage_path_for_world_valid_hash(settings_override):
    """Tests the internal _get_storage_path_for_world function with a valid hash."""
    default_test_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config_obj = settings_override.get_world_config(default_test_world_name)

    current_asset_storage_path = Path(world_config_obj.ASSET_STORAGE_PATH)
    test_hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    expected_path = current_asset_storage_path / "ab" / "cd" / test_hash
    assert file_storage._get_storage_path_for_world(test_hash, world_config_obj) == expected_path


def test_get_storage_path_for_world_short_hash_raises_value_error(settings_override):
    """Tests that _get_storage_path_for_world raises ValueError for short hashes."""
    default_test_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config_obj = settings_override.get_world_config(default_test_world_name)
    with pytest.raises(ValueError):
        file_storage._get_storage_path_for_world("abc", world_config_obj)


def test_store_file_creates_file_and_returns_hash(settings_override):
    """Tests storing a new file in the default test world."""
    default_test_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config_obj = settings_override.get_world_config(default_test_world_name)

    file_content = b"This is a test file."
    original_filename = "test.txt"
    expected_hash = hashlib.sha256(file_content).hexdigest()

    returned_hash = file_storage.store_file(file_content, world_config=world_config_obj, original_filename=original_filename)
    assert returned_hash == expected_hash

    expected_path = file_storage._get_storage_path_for_world(expected_hash, world_config_obj)
    assert expected_path.exists()
    assert expected_path.is_file()
    with open(expected_path, "rb") as f:
        assert f.read() == file_content


def test_store_file_empty_content(settings_override):
    """Tests storing a file with empty content in the default test world."""
    default_test_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config_obj = settings_override.get_world_config(default_test_world_name)

    file_content = b""
    original_filename = "empty.txt"
    expected_hash = hashlib.sha256(file_content).hexdigest()

    returned_hash = file_storage.store_file(file_content, world_config=world_config_obj, original_filename=original_filename)
    assert returned_hash == expected_hash

    expected_path = file_storage._get_storage_path_for_world(expected_hash, world_config_obj)
    assert expected_path.exists()
    with open(expected_path, "rb") as f:
        assert f.read() == file_content


def test_store_file_duplicate_content(settings_override):
    """Tests storing duplicate content in the default test world."""
    default_test_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config_obj = settings_override.get_world_config(default_test_world_name)

    file_content = b"Duplicate content."
    original_filename1 = "dup1.txt"
    original_filename2 = "dup2.txt"
    expected_hash = hashlib.sha256(file_content).hexdigest()

    hash1 = file_storage.store_file(file_content, world_config=world_config_obj, original_filename=original_filename1)
    assert hash1 == expected_hash
    path1 = file_storage._get_storage_path_for_world(hash1, world_config_obj)
    assert path1.exists()

    hash2 = file_storage.store_file(file_content, world_config=world_config_obj, original_filename=original_filename2)
    assert hash2 == expected_hash
    path2 = file_storage._get_storage_path_for_world(hash2, world_config_obj)
    assert path2 == path1

    with open(path1, "rb") as f:
        assert f.read() == file_content


def test_get_file_path_existing_file(settings_override):
    """Tests retrieving path for existing file in default test world."""
    default_test_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config_obj = settings_override.get_world_config(default_test_world_name)

    file_content = b"Find me."
    file_hash = file_storage.store_file(file_content, world_config=world_config_obj, original_filename="find_me.txt")

    retrieved_path = file_storage.get_file_path(file_hash, world_config=world_config_obj)
    expected_path = file_storage._get_storage_path_for_world(file_hash, world_config_obj).resolve()

    assert retrieved_path is not None
    assert retrieved_path == expected_path
    assert retrieved_path.exists()


def test_get_file_path_non_existing_file(settings_override):
    """Tests retrieving path for non-existing file in default test world."""
    default_test_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config_obj = settings_override.get_world_config(default_test_world_name) # Get world_config
    non_existent_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    retrieved_path = file_storage.get_file_path(non_existent_hash, world_config=world_config_obj) # Pass world_config
    assert retrieved_path is None


def test_get_file_path_invalid_identifier(settings_override):
    """Tests get_file_path with invalid identifier in default test world."""
    default_test_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config_obj = settings_override.get_world_config(default_test_world_name) # Get world_config
    assert file_storage.get_file_path("short", world_config=world_config_obj) is None # Pass world_config
    assert file_storage.get_file_path("", world_config=world_config_obj) is None # Pass world_config


def test_delete_file_existing(settings_override):
    """Tests deleting an existing file from default test world."""
    default_test_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config_obj = settings_override.get_world_config(default_test_world_name)

    file_content = b"To be deleted."
    file_hash = file_storage.store_file(file_content, world_config=world_config_obj, original_filename="delete_me.txt")
    file_path = file_storage.get_file_path(file_hash, world_config=world_config_obj) # Path before deletion
    assert file_path is not None and file_path.exists()

    parent_dir = file_path.parent
    grandparent_dir = parent_dir.parent
    assert parent_dir.exists() and grandparent_dir.exists()

    delete_result = file_storage.delete_file(file_hash, world_config=world_config_obj)
    assert delete_result is True
    assert file_storage.get_file_path(file_hash, world_config=world_config_obj) is None # Check it's gone
    assert not file_path.exists() # Original path should not exist

    assert not parent_dir.exists() # Empty parent dirs should be removed
    assert not grandparent_dir.exists()


def test_delete_file_non_existing(settings_override):
    """Tests deleting a non-existing file from default test world."""
    default_test_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config_obj = settings_override.get_world_config(default_test_world_name) # Get world_config
    non_existent_hash = "1111111111111111111111111111111111111111111111111111111111111111"
    delete_result = file_storage.delete_file(non_existent_hash, world_config=world_config_obj) # Pass world_config
    assert delete_result is False


def test_delete_file_leaves_non_empty_directories(settings_override):
    """Tests deleting a file doesn't remove parent dirs if not empty (default test world)."""
    default_test_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config_obj = settings_override.get_world_config(default_test_world_name)

    content1 = b"File 1 in shared dir"
    file_hash1 = file_storage.store_file(content1, world_config=world_config_obj, original_filename="file1.txt")

    # Construct path where file1's sub-sub-directory would be
    # This relies on _get_storage_path_for_world to be correct
    path_for_hash1_dir = file_storage._get_storage_path_for_world(file_hash1, world_config_obj).parent

    # Create a dummy file in the same directory to make it non-empty after file1 deletion
    dummy_file_in_subdir = path_for_hash1_dir / "dummy.txt"
    dummy_file_in_subdir.write_text("hello")
    assert dummy_file_in_subdir.exists()

    file_storage.delete_file(file_hash1, world_name=default_test_world)
    assert file_storage.get_file_path(file_hash1, world_name=default_test_world) is None

    assert path_for_hash1_dir.exists() # Parent dir of deleted file should remain
    assert dummy_file_in_subdir.exists() # Dummy file should still be there

    os.remove(dummy_file_in_subdir) # Clean up dummy
    # Try to remove dirs now they should be empty (or rmdir will fail, which is fine for test)
    try:
        os.rmdir(path_for_hash1_dir)
        os.rmdir(path_for_hash1_dir.parent)
    except OSError:
        pass

    # Base storage path for the world should still exist
    assert Path(world_config.ASSET_STORAGE_PATH).exists()
