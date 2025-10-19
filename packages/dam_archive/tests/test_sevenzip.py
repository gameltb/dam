"""Tests for 7z archive handling."""

import datetime
from pathlib import Path

import py7zr
import pytest

from dam_archive.base import to_stream_provider
from dam_archive.exceptions import InvalidPasswordError, UnsupportedArchiveError
from dam_archive.handlers.sevenzip import SevenZipArchiveHandler
from dam_archive.main import open_archive

CONTENT1 = b"content1"
NESTED_CONTENT = b"content_nested"


@pytest.fixture
def dummy_7z_file(tmp_path: Path) -> Path:
    """Create a dummy 7z file."""
    file_path = tmp_path / "test.7z"
    with py7zr.SevenZipFile(file_path, "w") as z:
        z.writestr(CONTENT1, "file1.txt")
    return file_path


@pytest.fixture
def protected_7z_file(tmp_path: Path) -> Path:
    """Create a password-protected 7z file."""
    file_path = tmp_path / "protected.7z"
    with py7zr.SevenZipFile(file_path, "w", password="password") as z:
        z.writestr(CONTENT1, "file1.txt")
    return file_path


@pytest.fixture
def nested_7z_file(tmp_path: Path) -> Path:
    """Create a 7z file with a nested file."""
    file_path = tmp_path / "nested.7z"
    with py7zr.SevenZipFile(file_path, "w") as z:
        z.writestr(NESTED_CONTENT, "folder/nested_file.txt")
    return file_path


@pytest.fixture
def regular_7z_archive(tmp_path: Path) -> Path:
    """Create a regular 7z archive for testing."""
    archive_path = tmp_path / "regular.7z"
    file_path = tmp_path / "file.txt"
    file_path.write_text("This is a test file for 7z archives.\n")
    with py7zr.SevenZipFile(archive_path, "w") as z:
        z.write(file_path, "file.txt")
    return archive_path


@pytest.fixture
def protected_7z_archive(tmp_path: Path) -> Path:
    """Create a protected 7z archive for testing."""
    archive_path = tmp_path / "protected.7z"
    file_path = tmp_path / "file.txt"
    file_path.write_text("This is a test file for 7z archives.\n")
    with py7zr.SevenZipFile(archive_path, "w", password="password") as z:
        z.write(file_path, "file.txt")
    return archive_path


@pytest.fixture
def bcj2_7z_archive(tmp_path: Path) -> Path:
    """Create a 7z archive with a BCJ2 filter for testing."""
    archive_path = tmp_path / "bcj2.7z"
    file_path = tmp_path / "hello_x86"
    file_path.write_bytes(b"\xe8\x00\x00\x00\x00")
    filters = [{"id": py7zr.FILTER_X86}]
    with py7zr.SevenZipFile(archive_path, "w", filters=filters) as z:
        z.write(file_path, "hello_x86")
    return archive_path


@pytest.mark.asyncio
async def test_open_7z_archive(dummy_7z_file: Path) -> None:
    """Test opening a simple 7z archive."""
    archive = None
    try:
        archive = await open_archive(dummy_7z_file, "application/x-7z-compressed")
        assert archive is not None
        files = archive.list_files()
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names
        for f_info in files:
            assert isinstance(f_info.modified_at, datetime.datetime)
            assert f_info.compressed_size is not None
            assert f_info.compressed_size > 0
    finally:
        if archive:
            await archive.close()


@pytest.mark.asyncio
async def test_unsupported_bcj2_archive_raises_error(bcj2_7z_archive: Path):
    """Test that SevenZipArchiveHandler raises UnsupportedArchiveError for BCJ2-filtered archives."""
    # We test SevenZipArchiveHandler directly, as open_archive would fall back to the CLI handler.
    stream_provider = to_stream_provider(bcj2_7z_archive)
    with pytest.raises(UnsupportedArchiveError):
        await SevenZipArchiveHandler.create(stream_provider)


@pytest.mark.asyncio
async def test_open_protected_7z_with_correct_password(protected_7z_file: Path) -> None:
    """Test opening a protected 7z archive with the correct password."""
    archive = None
    try:
        archive = await open_archive(protected_7z_file, "application/x-7z-compressed", password="password")
        assert archive is not None
        member_info, f_in_zip = archive.open_file("file1.txt")
        with f_in_zip:
            assert f_in_zip.read() == CONTENT1
        assert member_info.name == "file1.txt"
    finally:
        if archive:
            await archive.close()


@pytest.mark.asyncio
async def test_open_protected_7z_with_incorrect_password(protected_7z_file: Path) -> None:
    """Test opening a protected 7z archive with an incorrect password."""
    archive = None
    try:
        with pytest.raises(InvalidPasswordError):
            archive = await open_archive(protected_7z_file, "application/x-7z-compressed", password="wrong_password")
    finally:
        if archive:
            await archive.close()


@pytest.mark.asyncio
async def test_open_nested_7z_file(nested_7z_file: Path) -> None:
    """Test opening a 7z file with a nested file."""
    archive = None
    try:
        archive = await open_archive(nested_7z_file, "application/x-7z-compressed")
        assert archive is not None
        files = archive.list_files()
        assert "folder/nested_file.txt" in [m.name for m in files]
        member_info, f_in_zip = archive.open_file("folder/nested_file.txt")
        with f_in_zip:
            assert f_in_zip.read() == NESTED_CONTENT
        assert member_info.name == "folder/nested_file.txt"
    finally:
        if archive:
            await archive.close()


@pytest.mark.asyncio
async def test_iter_files_7z_archive(dummy_7z_file: Path) -> None:
    """Test iterating over files in a 7z archive."""
    archive = None
    try:
        archive = await open_archive(dummy_7z_file, "application/x-7z-compressed")
        assert archive is not None

        files = list(archive.iter_files())
        assert len(files) == 1

        member_info, member_stream = files[0]
        assert member_info.name == "file1.txt"
        assert member_info.size == len(CONTENT1)

        with member_stream as f_in_zip:
            assert f_in_zip.read() == CONTENT1
    finally:
        if archive:
            await archive.close()
