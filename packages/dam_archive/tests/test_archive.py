import zipfile
from pathlib import Path

import pytest

from dam_archive.main import open_archive


@pytest.fixture
def dummy_zip_file(tmp_path: Path) -> Path:
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file1.txt", b"content1")
        zf.writestr("dir/file2.txt", b"content2")
    return zip_path


def test_open_zip_archive(dummy_zip_file: Path):
    archive = open_archive(str(dummy_zip_file))
    assert archive is not None
    files = archive.list_files()
    assert "file1.txt" in files
    assert "dir/file2.txt" in files

    with archive.open_file("file1.txt") as f:
        assert f.read() == b"content1"

    with archive.open_file("dir/file2.txt") as f:
        assert f.read() == b"content2"


@pytest.fixture
def nested_zip_file(tmp_path: Path) -> Path:
    zip_path = tmp_path / "nested.zip"

    # Create the inner zip first
    inner_zip_path = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner_zip_path, "w") as zf:
        zf.writestr("file2.txt", b"content2")

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file1.txt", b"content1")
        zf.write(inner_zip_path, "inner.zip")

    return zip_path


def test_open_nested_zip_archive(nested_zip_file: Path):
    archive = open_archive(str(nested_zip_file))
    assert archive is not None
    with archive.open_file("inner.zip/file2.txt") as f:
        assert f.read() == b"content2"
