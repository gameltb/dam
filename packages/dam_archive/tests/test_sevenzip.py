from pathlib import Path

import py7zr
import pytest

from dam_archive.main import open_archive


@pytest.fixture
def dummy_7z_file(tmp_path: Path) -> Path:
    sevenzip_path = tmp_path / "test.7z"
    with py7zr.SevenZipFile(sevenzip_path, "w") as zf:
        zf.writestr(b"content1", "file1.txt")
        zf.writestr(b"content2", "dir/file2.txt")
    return sevenzip_path


def test_open_7z_archive(dummy_7z_file: Path) -> None:
    archive = open_archive(str(dummy_7z_file), dummy_7z_file.name)
    assert archive is not None
    files = archive.list_files()
    assert "file1.txt" in files
    assert "dir/file2.txt" in files

    with archive.open_file("file1.txt") as f:
        assert f.read() == b"content1"

    with archive.open_file("dir/file2.txt") as f:
        assert f.read() == b"content2"


@pytest.fixture
def protected_7z_file(tmp_path: Path) -> Path:
    sevenzip_path = tmp_path / "protected.7z"
    with py7zr.SevenZipFile(sevenzip_path, "w", password="password") as zf:
        zf.writestr(b"content1", "file1.txt")
    return sevenzip_path


def test_open_protected_7z_archive(protected_7z_file: Path) -> None:
    archive = open_archive(str(protected_7z_file), protected_7z_file.name, passwords=["password"])
    assert archive is not None
    with archive.open_file("file1.txt") as f:
        assert f.read() == b"content1"
