"""Configuration for pytest."""

from pathlib import Path

import pytest

from dam_archive.handlers.sevenzip_cli import SevenZipCliArchiveHandler

pytest_plugins = ["dam_test_utils.fixtures"]


@pytest.fixture
def regular_7z_archive(tmp_path: Path) -> Path:
    """Create a regular 7z archive for testing."""
    archive_path = tmp_path / "regular.7z"
    file_path = tmp_path / "file.txt"
    file_path.write_text("This is a test file for 7z archives.\n")
    SevenZipCliArchiveHandler.run_7z_static(["a", str(archive_path), str(file_path)])
    return archive_path


@pytest.fixture
def protected_7z_archive(tmp_path: Path) -> Path:
    """Create a protected 7z archive for testing."""
    archive_path = tmp_path / "protected.7z"
    file_path = tmp_path / "file.txt"
    file_path.write_text("This is a test file for 7z archives.\n")
    SevenZipCliArchiveHandler.run_7z_static(["a", str(archive_path), str(file_path), "-ppassword"])
    return archive_path


@pytest.fixture
def bcj2_7z_archive(tmp_path: Path) -> Path:
    """Create a 7z archive with a BCJ2 filter for testing."""
    archive_path = tmp_path / "bcj2.7z"
    file_path = tmp_path / "hello_x86"
    file_path.write_bytes(b"\xe8\x00\x00\x00\x00")
    SevenZipCliArchiveHandler.run_7z_static(["a", str(archive_path), str(file_path)])
    return archive_path
