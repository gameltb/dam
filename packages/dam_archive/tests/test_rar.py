from pathlib import Path

import pytest

from dam_archive.main import open_archive

# Note: These tests are skipped because there is no reliable way to create .rar files programmatically in Python.
# The `rarfile` library can only read .rar files, not create them.
# To run these tests, you would need to add pre-existing .rar files to the test assets.


@pytest.mark.skip(reason="Cannot create .rar files programmatically.")
@pytest.fixture
def dummy_rar_file(tmp_path: Path) -> Path:
    # This is a placeholder and will not work.
    rar_path = tmp_path / "test.rar"
    with open(rar_path, "wb") as f:
        f.write(b"dummy content")
    return rar_path


@pytest.mark.skip(reason="Cannot create .rar files programmatically.")
def test_open_rar_archive(dummy_rar_file: Path) -> None:
    archive = open_archive(str(dummy_rar_file), dummy_rar_file.name)
    assert archive is not None
    files = archive.list_files()
    file_names = [f.name for f in files]
    assert "file1.txt" in file_names
    assert "dir/file2.txt" in file_names

    member_info, f = archive.open_file("file1.txt")
    with f:
        assert f.read() == b"content1"
    assert member_info.name == "file1.txt"

    member_info, f = archive.open_file("dir/file2.txt")
    with f:
        assert f.read() == b"content2"
    assert member_info.name == "dir/file2.txt"


@pytest.mark.skip(reason="Cannot create .rar files programmatically.")
@pytest.fixture
def protected_rar_file(tmp_path: Path) -> Path:
    # This is a placeholder and will not work.
    rar_path = tmp_path / "protected.rar"
    with open(rar_path, "wb") as f:
        f.write(b"dummy content")
    return rar_path


@pytest.mark.skip(reason="Cannot create .rar files programmatically.")
def test_open_protected_rar_archive(protected_rar_file: Path) -> None:
    archive = open_archive(str(protected_rar_file), protected_rar_file.name, password="password")
    assert archive is not None
    member_info, f = archive.open_file("file1.txt")
    with f:
        assert f.read() == b"content1"
    assert member_info.name == "file1.txt"


@pytest.mark.skip(reason="Cannot create .rar files programmatically.")
def test_iter_files_rar_archive(dummy_rar_file: Path) -> None:
    # This test checks the iter_files method for rar archives.
    archive = open_archive(str(dummy_rar_file), dummy_rar_file.name)
    assert archive is not None

    files = list(archive.iter_files())
    assert len(files) == 2  # Assuming dummy rar has 2 files
    for member_info, f in files:
        with f:
            assert f.read()
        assert member_info.name
        assert member_info.size >= 0

    # Further assertions would go here if the test could be run
