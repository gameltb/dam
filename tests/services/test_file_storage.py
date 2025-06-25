import hashlib
import os
import shutil
from pathlib import Path

import pytest

from dam.core.config import settings
from dam.services import file_storage

# Use a temporary directory for tests that overrides the default settings.ASSET_STORAGE_PATH
TEST_STORAGE_BASE_PATH = Path("./test_dam_storage")


@pytest.fixture(autouse=True)
def manage_test_storage_directory(monkeypatch):
    """
    Fixture to create and clean up the test storage directory before and after each test.
    Also, it patches settings.ASSET_STORAGE_PATH to use the test directory.
    """
    # Patch the ASSET_STORAGE_PATH for the duration of the test
    monkeypatch.setattr(settings, "ASSET_STORAGE_PATH", str(TEST_STORAGE_BASE_PATH))

    # Teardown: Clean up the directory after tests
    if TEST_STORAGE_BASE_PATH.exists():
        shutil.rmtree(TEST_STORAGE_BASE_PATH)
    TEST_STORAGE_BASE_PATH.mkdir(parents=True, exist_ok=True)

    yield  # Test runs here

    # Teardown: Clean up the directory after tests
    if TEST_STORAGE_BASE_PATH.exists():
        shutil.rmtree(TEST_STORAGE_BASE_PATH)


def test_get_storage_path_valid_hash():
    """Tests the internal _get_storage_path function with a valid hash."""
    test_hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    expected_path = TEST_STORAGE_BASE_PATH / "ab" / "cd" / test_hash
    assert file_storage._get_storage_path(test_hash) == expected_path


def test_get_storage_path_short_hash_raises_value_error():
    """Tests that _get_storage_path raises ValueError for short hashes."""
    with pytest.raises(ValueError):
        file_storage._get_storage_path("abc")


def test_store_file_creates_file_and_returns_hash():
    """Tests storing a new file."""
    file_content = b"This is a test file."
    original_filename = "test.txt"
    expected_hash = hashlib.sha256(file_content).hexdigest()

    returned_hash = file_storage.store_file(file_content, original_filename)
    assert returned_hash == expected_hash

    expected_path = file_storage._get_storage_path(expected_hash)
    assert expected_path.exists()
    assert expected_path.is_file()
    with open(expected_path, "rb") as f:
        assert f.read() == file_content


def test_store_file_empty_content():
    """Tests storing a file with empty content."""
    file_content = b""
    original_filename = "empty.txt"
    expected_hash = hashlib.sha256(file_content).hexdigest()

    returned_hash = file_storage.store_file(file_content, original_filename)
    assert returned_hash == expected_hash

    expected_path = file_storage._get_storage_path(expected_hash)
    assert expected_path.exists()
    with open(expected_path, "rb") as f:
        assert f.read() == file_content


def test_store_file_duplicate_content():
    """Tests that storing the same content multiple times doesn't error and uses the same storage."""
    file_content = b"Duplicate content."
    original_filename1 = "dup1.txt"
    original_filename2 = "dup2.txt"
    expected_hash = hashlib.sha256(file_content).hexdigest()

    hash1 = file_storage.store_file(file_content, original_filename1)
    assert hash1 == expected_hash
    path1 = file_storage._get_storage_path(hash1)
    assert path1.exists()

    # Store again
    hash2 = file_storage.store_file(file_content, original_filename2)
    assert hash2 == expected_hash
    path2 = file_storage._get_storage_path(hash2)
    assert path2 == path1  # Should resolve to the exact same path

    # Check that the file was written only once (e.g. by checking mtime or by ensuring no error on second write)
    # The implementation overwrites if it exists, which is fine for content-addressable.
    # More importantly, it doesn't create a new file or change the hash.
    with open(path1, "rb") as f:
        assert f.read() == file_content


def test_get_file_path_existing_file():
    """Tests retrieving the path for an existing file."""
    file_content = b"Find me."
    file_hash = file_storage.store_file(file_content, "find_me.txt")

    retrieved_path = file_storage.get_file_path(file_hash)
    expected_path = file_storage._get_storage_path(file_hash).resolve()

    assert retrieved_path is not None
    assert retrieved_path == expected_path
    assert retrieved_path.exists()


def test_get_file_path_non_existing_file():
    """Tests retrieving the path for a file that hasn't been stored."""
    non_existent_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    retrieved_path = file_storage.get_file_path(non_existent_hash)
    assert retrieved_path is None


def test_get_file_path_invalid_identifier():
    """Tests get_file_path with an invalid (e.g., too short) identifier."""
    assert file_storage.get_file_path("short") is None
    assert file_storage.get_file_path("") is None


def test_delete_file_existing():
    """Tests deleting an existing file."""
    file_content = b"To be deleted."
    file_hash = file_storage.store_file(file_content, "delete_me.txt")
    file_path = file_storage.get_file_path(file_hash)
    assert file_path is not None
    assert file_path.exists()

    # Verify directory structure exists before deletion
    parent_dir = file_path.parent
    grandparent_dir = parent_dir.parent
    assert parent_dir.exists()
    assert grandparent_dir.exists()

    delete_result = file_storage.delete_file(file_hash)
    assert delete_result is True
    assert file_storage.get_file_path(file_hash) is None
    assert not file_path.exists()

    # Check if directories were removed (they should be if they became empty)
    assert not parent_dir.exists()
    assert not grandparent_dir.exists()


def test_delete_file_non_existing():
    """Tests deleting a non-existing file."""
    non_existent_hash = "1111111111111111111111111111111111111111111111111111111111111111"
    delete_result = file_storage.delete_file(non_existent_hash)
    assert delete_result is False


def test_delete_file_leaves_non_empty_directories():
    """Tests that deleting a file does not remove parent dirs if they are not empty."""
    content1 = b"File 1 in shared dir"
    # The variable 'content2' was previously defined here but not used.
    # b"File 2 in shared dir"  # different content, different hash, but could be forced into same subdirs for testing

    # To ensure they go into the same sub-sub-directory, we'd need to craft hashes.
    # Easier: store two files, then delete one. The other one's dir should remain.

    # The variable 'hash1' was previously defined here but not used.
    # hashlib.sha256(content1).hexdigest()  # e.g., 'a1...'
    # Hash for content2, engineered to share subdirs if possible, or just rely on _get_storage_path
    # For simplicity, let's ensure they are different enough to be distinct files.
    # If hash1 = "aabbccdd...", hash2 = "aabbeeff..." they would share ./aa/bb/
    # This test is simpler: just ensure the base ASSET_STORAGE_PATH is not deleted.

    file_hash1 = file_storage.store_file(content1, "file1.txt")
    # The variable 'path1' was previously defined here but not used.
    # file_storage.get_file_path(file_hash1)

    # Create another file that will reside in a *different* sub-sub-directory
    # to ensure the top-level test storage path itself isn't removed.
    # Or, more simply, one that ensures that if file1 is in aa/bb/hash1,
    # and file2 is in aa/cc/hash2, deleting file1 doesn't remove "aa".

    # To make them share the first level dir (e.g. "ab") but not the second (e.g. "cd" vs "ce")
    # we need to find/craft hashes. This is too complex for this test.
    # The current delete logic tries to remove file.parent and file.parent.parent.
    # If two files are in same parent.parent, e.g. ...BASE_PATH / "ab" / "cd" / hash1
    # and ...BASE_PATH / "ab" / "cd" / hash2
    # then deleting hash1 should remove "hash1", but not "cd" or "ab" because "hash2" is there.

    # Let's create two files that will share the same subdirectories.
    # We can't easily force hashes, so let's assume store_file works.
    # The test for directory deletion logic is more about whether os.rmdir fails gracefully.

    # The variable 'file_content_sibling' was previously defined here but not used.
    # b"Sibling file in the same sub-sub-directory, different name"
    # The variable 'sibling_hash_prefix' was previously defined here but not used.
    # file_hash1[:4]  # e.g. "abcd"

    # This is tricky: we need another file in the same sub_dir_1/sub_dir_2 path.
    # The easiest way is to store another file and hope for a collision in the first 4 chars (unlikely)
    # or manually create a structure.

    # Let's simplify: The `delete_file` only removes parent and grandparent if `os.rmdir` succeeds (dir is empty).
    # So, if a grandparent directory `settings.ASSET_STORAGE_PATH / hash_prefix1 / hash_prefix2`
    # contains another file (or directory), it won't be removed.

    base_path_for_hash1 = file_storage._get_storage_path(file_hash1).parent  # .../ab/cd/
    # Create a dummy file in the same directory as the stored file's directory
    with open(base_path_for_hash1 / "dummy.txt", "w") as f:
        f.write("hello")

    assert (base_path_for_hash1 / "dummy.txt").exists()

    file_storage.delete_file(file_hash1)
    assert not file_storage.get_file_path(file_hash1)  # File itself is gone

    # The parent directory of the deleted file should still exist because of dummy.txt
    assert base_path_for_hash1.exists()
    assert (base_path_for_hash1 / "dummy.txt").exists()  # dummy file still there

    # Clean up the dummy
    os.remove(base_path_for_hash1 / "dummy.txt")
    # Now try to remove the dirs, they should be empty now
    try:
        os.rmdir(base_path_for_hash1)  # .../ab/cd/
        os.rmdir(base_path_for_hash1.parent)  # .../ab/
    except OSError:
        pass  # This is fine if they were already removed or other test cleanup did it

    # Final check: the base test storage path should always exist
    assert TEST_STORAGE_BASE_PATH.exists()
