import datetime
from pathlib import Path

import py7zr
import pytest

from dam_archive.base import to_stream_provider
from dam_archive.exceptions import InvalidPasswordError, UnsupportedArchiveError
from dam_archive.handlers.sevenzip import SevenZipArchiveHandler
from dam_archive.main import open_archive


@pytest.fixture
def dummy_7z_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "test.7z"
    with py7zr.SevenZipFile(file_path, "w") as z:
        z.writestr(b"content1", "file1.txt")
    return file_path


def test_open_7z_archive(dummy_7z_file: Path) -> None:
    archive = None
    try:
        with open(dummy_7z_file, "rb") as f:
            archive = open_archive(f, "application/x-7z-compressed")
            assert archive is not None
            files = archive.list_files()
            file_names = [f.name for f in files]
            assert "file1.txt" in file_names
            for f in files:
                assert isinstance(f.modified_at, datetime.datetime)
    finally:
        if archive:
            archive.close()


def test_unsupported_bcj2_archive_raises_error(bcj2_7z_archive: Path):
    """
    Tests that SevenZipArchiveHandler raises UnsupportedArchiveError for BCJ2-filtered archives.
    """
    # We test SevenZipArchiveHandler directly, as open_archive would fall back to the CLI handler.
    with open(bcj2_7z_archive, "rb") as f:
        stream_provider = to_stream_provider(f)
        with pytest.raises(UnsupportedArchiveError):
            SevenZipArchiveHandler(stream_provider)


@pytest.fixture
def protected_7z_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "protected.7z"
    with py7zr.SevenZipFile(file_path, "w", password="password") as z:
        z.writestr(b"content1", "file1.txt")
    return file_path


def test_open_protected_7z_with_correct_password(protected_7z_file: Path) -> None:
    archive = None
    try:
        with open(protected_7z_file, "rb") as f:
            archive = open_archive(f, "application/x-7z-compressed", password="password")
            assert archive is not None
            member_info, f_in_zip = archive.open_file("file1.txt")
            with f_in_zip:
                assert f_in_zip.read() == b"content1"
            assert member_info.name == "file1.txt"
    finally:
        if archive:
            archive.close()


def test_open_protected_7z_with_incorrect_password(protected_7z_file: Path) -> None:
    archive = None
    try:
        with open(protected_7z_file, "rb") as f:
            with pytest.raises(InvalidPasswordError):
                archive = open_archive(f, "application/x-7z-compressed", password="wrong_password")
    finally:
        if archive:
            archive.close()


@pytest.fixture
def nested_7z_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "nested.7z"
    with py7zr.SevenZipFile(file_path, "w") as z:
        z.writestr(b"content_nested", "folder/nested_file.txt")
    return file_path


def test_open_nested_7z_file(nested_7z_file: Path) -> None:
    archive = None
    try:
        with open(nested_7z_file, "rb") as f:
            archive = open_archive(f, "application/x-7z-compressed")
            assert archive is not None
            files = archive.list_files()
            assert "folder/nested_file.txt" in [m.name for m in files]
            member_info, f_in_zip = archive.open_file("folder/nested_file.txt")
            with f_in_zip:
                assert f_in_zip.read() == b"content_nested"
            assert member_info.name == "folder/nested_file.txt"
    finally:
        if archive:
            archive.close()


def test_iter_files_7z_archive(dummy_7z_file: Path) -> None:
    archive = None
    try:
        with open(dummy_7z_file, "rb") as f:
            archive = open_archive(f, "application/x-7z-compressed")
            assert archive is not None

            files = list(archive.iter_files())
            assert len(files) == 1

            member_info, member_stream = files[0]
            assert member_info.name == "file1.txt"
            assert member_info.size == 8

            with member_stream as f_in_zip:
                assert f_in_zip.read() == b"content1"
    finally:
        if archive:
            archive.close()
