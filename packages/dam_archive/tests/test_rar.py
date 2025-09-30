import subprocess
from pathlib import Path

import pytest

from dam_archive.main import open_archive


@pytest.fixture
def dummy_rar_file(tmp_path: Path) -> Path:
    """Creates a dummy rar file with a couple of files in it."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "file1.txt").write_text("content1")
    (src_dir / "dir").mkdir()
    (src_dir / "dir" / "file2.txt").write_text("content2")

    rar_path = tmp_path / "test.rar"

    # Use the rar command line tool to create the archive.
    subprocess.run(
        ["rar", "a", str(rar_path), "file1.txt", "dir/file2.txt"],
        cwd=src_dir,
        check=True,
        capture_output=True,
    )
    return rar_path


@pytest.mark.asyncio
async def test_open_rar_archive(dummy_rar_file: Path) -> None:
    archive = await open_archive(str(dummy_rar_file), "application/vnd.rar")
    assert archive is not None
    files = archive.list_files()
    file_names = [f.name for f in files]
    assert "file1.txt" in file_names
    assert "dir/file2.txt" in file_names

    member_info, f = archive.open_file("file1.txt")
    with f:
        assert f.read() == b"content1"
    assert member_info.name == "file1.txt"

    member_info, f = archive.open_file("dir/file2.txt")
    with f:
        assert f.read() == b"content2"
    assert member_info.name == "dir/file2.txt"
    await archive.close()


@pytest.fixture
def protected_rar_file(tmp_path: Path) -> Path:
    """Creates a dummy password-protected rar file."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "file1.txt").write_text("content1")

    rar_path = tmp_path / "protected.rar"

    # Use the rar command line tool to create the archive.
    subprocess.run(
        ["rar", "a", "-ppassword", str(rar_path), "file1.txt"],
        cwd=src_dir,
        check=True,
        capture_output=True,
    )
    return rar_path


@pytest.mark.asyncio
async def test_open_protected_rar_archive(protected_rar_file: Path) -> None:
    archive = await open_archive(str(protected_rar_file), "application/vnd.rar", password="password")
    assert archive is not None
    member_info, f = archive.open_file("file1.txt")
    with f:
        assert f.read() == b"content1"
    assert member_info.name == "file1.txt"
    await archive.close()


@pytest.mark.asyncio
async def test_iter_files_rar_archive(dummy_rar_file: Path) -> None:
    # This test checks the iter_files method for rar archives.
    archive = await open_archive(str(dummy_rar_file), "application/vnd.rar")
    assert archive is not None

    files = list(archive.iter_files())
    assert len(files) == 2
    for member_info, f in files:
        with f:
            assert f.read()
        assert member_info.name
        assert member_info.size >= 0

    await archive.close()
