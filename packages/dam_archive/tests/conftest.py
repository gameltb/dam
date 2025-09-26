import subprocess
import zipfile
from pathlib import Path

import pytest
from dam.core import World
from dam_fs.plugin import FsPlugin
from pytest import TempPathFactory

from dam_archive.plugin import ArchivePlugin

pytest_plugins = ["dam_test_utils.fixtures"]


@pytest.fixture(scope="function", autouse=True)
def setup_world_with_plugins(test_world_alpha: World):
    """Automatically sets up the test world with necessary plugins."""
    test_world_alpha.add_plugin(FsPlugin())
    test_world_alpha.add_plugin(ArchivePlugin())


@pytest.fixture(scope="session")
def test_7z_content(tmp_path_factory: TempPathFactory) -> Path:
    """Creates a directory with content for 7z archives."""
    content_dir = tmp_path_factory.mktemp("test_7z_content")
    (content_dir / "file.txt").write_text("This is a test file for 7z archives.\n")

    # Create a dummy file for BCJ2 filter testing
    (content_dir / "hello_x86").write_bytes(b"dummy executable content for bcj2 test")

    return content_dir


@pytest.fixture(scope="session")
def regular_7z_archive(tmp_path_factory: TempPathFactory, test_7z_content: Path) -> Path:
    """Creates a regular 7z archive."""
    archive_path = tmp_path_factory.mktemp("archives") / "regular_archive.7z"
    subprocess.run(
        ["7z", "a", str(archive_path), "file.txt"],
        cwd=test_7z_content,
        check=True,
    )
    return archive_path


@pytest.fixture(scope="session")
def protected_7z_archive(tmp_path_factory: TempPathFactory, test_7z_content: Path) -> Path:
    """Creates a password-protected 7z archive."""
    archive_path = tmp_path_factory.mktemp("archives") / "protected_archive.7z"
    subprocess.run(
        ["7z", "a", "-ppassword", str(archive_path), "file.txt"],
        cwd=test_7z_content,
        check=True,
    )
    return archive_path


@pytest.fixture(scope="session")
def bcj2_7z_archive(tmp_path_factory: TempPathFactory, test_7z_content: Path) -> Path:
    """Creates a BCJ2-filtered 7z archive."""
    archive_path = tmp_path_factory.mktemp("archives") / "bcj2_archive.7z"
    subprocess.run(
        ["7z", "a", "-m0=BCJ2", "-m1=LZMA", str(archive_path), "hello_x86"],
        cwd=test_7z_content,
        check=True,
    )
    return archive_path


@pytest.fixture
def test_archives(tmp_path: Path) -> tuple[Path, Path]:
    # Regular archive
    regular_archive_path = tmp_path / "regular.zip"
    with zipfile.ZipFile(regular_archive_path, "w") as zf:
        zf.writestr("file1.txt", "file one")
        zf.writestr("file2.txt", "file two")
        zf.comment = b"regular archive comment"

    # Protected archive
    protected_archive_path = tmp_path / "protected.zip"
    with zipfile.ZipFile(protected_archive_path, "w") as zf:
        zf.setpassword(b"password")
        zf.writestr("file1.txt", "file one")
        zf.writestr("file2.txt", "file two")
        zf.comment = b"protected archive comment"

    return regular_archive_path, protected_archive_path
