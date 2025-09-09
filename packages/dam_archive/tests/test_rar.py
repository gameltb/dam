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
    assert "file1.txt" in files
    assert "dir/file2.txt" in files

    with archive.open_file("file1.txt") as f:
        assert f.read() == b"content1"

    with archive.open_file("dir/file2.txt") as f:
        assert f.read() == b"content2"


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
    archive = open_archive(str(protected_rar_file), protected_rar_file.name, passwords=["password"])
    assert archive is not None
    with archive.open_file("file1.txt") as f:
        assert f.read() == b"content1"
