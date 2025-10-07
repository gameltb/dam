"""Tests for the archive opening logic."""

import datetime
import subprocess
import zipfile
from pathlib import Path

import pytest

from dam_archive.base import to_stream_provider
from dam_archive.exceptions import InvalidPasswordError
from dam_archive.main import open_archive

CONTENT1 = b"content1"
CONTENT2 = b"content2"


@pytest.fixture
def dummy_zip_file(tmp_path: Path) -> Path:
    """Create a dummy zip file for testing."""
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file1.txt", CONTENT1)
        zf.writestr("dir/file2.txt", CONTENT2)
        zf.comment = b"test comment"
    return zip_path


@pytest.mark.asyncio
async def test_open_zip_archive(dummy_zip_file: Path) -> None:
    """Test opening a simple zip archive."""
    archive = await open_archive(dummy_zip_file, "application/zip")
    assert archive is not None
    assert archive.comment == "test comment"
    files = archive.list_files()
    file_names = [f.name for f in files]
    assert "file1.txt" in file_names
    assert "dir/file2.txt" in file_names

    for f_info in files:
        assert isinstance(f_info.modified_at, datetime.datetime)

    member_info, f_in_zip = archive.open_file("file1.txt")
    with f_in_zip:
        assert f_in_zip.read() == CONTENT1
    assert member_info.name == "file1.txt"
    assert member_info.size == len(CONTENT1)

    member_info, f_in_zip = archive.open_file("dir/file2.txt")
    with f_in_zip:
        assert f_in_zip.read() == CONTENT2
    assert member_info.name == "dir/file2.txt"
    assert member_info.size == len(CONTENT2)
    await archive.close()


@pytest.fixture
def protected_zip_file(tmp_path: Path) -> Path:
    """Create a password-protected zip file for testing."""
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


@pytest.mark.asyncio
async def test_open_protected_zip_with_correct_password(protected_zip_file: Path) -> None:
    """Test opening a protected zip archive with the correct password."""
    archive = await open_archive(protected_zip_file, "application/zip", password="password")
    assert archive is not None
    member_info, f_in_zip = archive.open_file("file1.txt")
    with f_in_zip:
        assert f_in_zip.read() == b"content1"
    assert member_info.name == "file1.txt"
    await archive.close()


@pytest.mark.asyncio
async def test_open_protected_zip_with_incorrect_password(protected_zip_file: Path) -> None:
    """Test opening a protected zip archive with an incorrect password."""
    with pytest.raises(InvalidPasswordError):
        await open_archive(protected_zip_file, "application/zip", password="wrong_password")


@pytest.mark.asyncio
async def test_open_zip_archive_with_stream_provider(dummy_zip_file: Path) -> None:
    """Test opening a zip archive using a stream provider."""
    stream_provider = to_stream_provider(str(dummy_zip_file))
    archive = await open_archive(stream_provider, "application/zip")
    assert archive is not None
    assert archive.comment == "test comment"
    files = archive.list_files()
    file_names = [f.name for f in files]
    assert "file1.txt" in file_names
    assert "dir/file2.txt" in file_names

    for f_info in files:
        assert isinstance(f_info.modified_at, datetime.datetime)

    member_info, f_in_zip = archive.open_file("file1.txt")
    with f_in_zip:
        assert f_in_zip.read() == CONTENT1
    assert member_info.name == "file1.txt"
    assert member_info.size == len(CONTENT1)

    member_info, f_in_zip = archive.open_file("dir/file2.txt")
    with f_in_zip:
        assert f_in_zip.read() == CONTENT2
    assert member_info.name == "dir/file2.txt"
    assert member_info.size == len(CONTENT2)
    await archive.close()
