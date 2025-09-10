from pathlib import Path

import py7zr
import pytest

from dam_archive.exceptions import InvalidPasswordError
from dam_archive.main import open_archive


@pytest.fixture
def dummy_7z_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "test.7z"
    with py7zr.SevenZipFile(file_path, "w") as z:
        z.writestr(b"content1", "file1.txt")
    return file_path


def test_open_7z_archive(dummy_7z_file: Path) -> None:
    with open(dummy_7z_file, "rb") as f:
        archive = open_archive(f, dummy_7z_file.name)
        assert archive is not None
        files = archive.list_files()
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names


@pytest.fixture
def protected_7z_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "protected.7z"
    with py7zr.SevenZipFile(file_path, "w", password="password") as z:
        z.writestr(b"content1", "file1.txt")
    return file_path


def test_open_protected_7z_with_correct_password(protected_7z_file: Path) -> None:
    with open(protected_7z_file, "rb") as f:
        archive = open_archive(f, protected_7z_file.name, password="password")
        assert archive is not None
        with archive.open_file("file1.txt") as f_in_zip:
            assert f_in_zip.read() == b"content1"


def test_open_protected_7z_with_incorrect_password(protected_7z_file: Path) -> None:
    with open(protected_7z_file, "rb") as f:
        with pytest.raises(InvalidPasswordError):
            open_archive(f, protected_7z_file.name, password="wrong_password")


@pytest.fixture
def nested_7z_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "nested.7z"
    with py7zr.SevenZipFile(file_path, "w") as z:
        z.writestr(b"content_nested", "folder/nested_file.txt")
    return file_path


def test_open_nested_7z_file(nested_7z_file: Path) -> None:
    with open(nested_7z_file, "rb") as f:
        archive = open_archive(f, nested_7z_file.name)
        assert archive is not None
        files = archive.list_files()
        assert "folder/nested_file.txt" in [m.name for m in files]
        with archive.open_file("folder/nested_file.txt") as f_in_zip:
            assert f_in_zip.read() == b"content_nested"
