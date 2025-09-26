import datetime
import subprocess
import zipfile
from pathlib import Path

import pytest

from dam_archive.exceptions import InvalidPasswordError
from dam_archive.main import open_archive


@pytest.fixture
def dummy_zip_file(tmp_path: Path) -> Path:
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file1.txt", b"content1")
        zf.writestr("dir/file2.txt", b"content2")
        zf.comment = b"test comment"
    return zip_path


def test_open_zip_archive(dummy_zip_file: Path) -> None:
    with open(dummy_zip_file, "rb") as f:
        archive = open_archive(f, "application/zip")
        assert archive is not None
        assert archive.comment == "test comment"
        files = archive.list_files()
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names
        assert "dir/file2.txt" in file_names

        for f in files:
            assert isinstance(f.modified_at, datetime.datetime)

        member_info, f_in_zip = archive.open_file("file1.txt")
        with f_in_zip:
            assert f_in_zip.read() == b"content1"
        assert member_info.name == "file1.txt"
        assert member_info.size == 8

        member_info, f_in_zip = archive.open_file("dir/file2.txt")
        with f_in_zip:
            assert f_in_zip.read() == b"content2"
        assert member_info.name == "dir/file2.txt"
        assert member_info.size == 8


@pytest.fixture
def protected_zip_file(tmp_path: Path) -> Path:
    zip_path = tmp_path / "protected.zip"
    file_to_zip = tmp_path / "file1.txt"
    file_to_zip.write_text("content1")

    subprocess.run(
        [
            "zip",
            "-j",
            "-P",
            "password",
            str(zip_path),
            str(file_to_zip),
        ],
        check=True,
    )

    return zip_path


def test_open_protected_zip_with_correct_password(protected_zip_file: Path) -> None:
    with open(protected_zip_file, "rb") as f:
        archive = open_archive(f, "application/zip", password="password")
        assert archive is not None
        member_info, f_in_zip = archive.open_file("file1.txt")
        with f_in_zip:
            assert f_in_zip.read() == b"content1"
        assert member_info.name == "file1.txt"


def test_open_protected_zip_with_incorrect_password(protected_zip_file: Path) -> None:
    with open(protected_zip_file, "rb") as f:
        with pytest.raises(InvalidPasswordError):
            open_archive(f, "application/zip", password="wrong_password")
