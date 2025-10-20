"""Tests for the asset verification CLI command."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import aiofiles
import pytest
from dam.commands.analysis_commands import AutoSetMimeTypeCommand
from dam_archive.commands import IngestArchiveCommand
from dam_archive.settings import ArchiveSettingsComponent
from dam_fs.commands import RegisterLocalFileCommand
from dam_fs.settings import FsSettingsComponent
from dam_test_utils.types import WorldFactory

from dam_app.cli import verify_assets_logic


async def _get_sha256(file_path: Path) -> str:
    """Calculate the SHA256 hash of a file."""
    h = hashlib.sha256()
    async with aiofiles.open(file_path, "rb") as f:
        while True:
            data = await f.read(65536)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


@pytest.mark.asyncio
async def test_verify_single_file_ok(
    tmp_path: Path,
    world_factory: WorldFactory,
):
    """Test that a single, unmodified file passes verification."""
    # Create a test world
    world = await world_factory(
        "test_world",
        [
            FsSettingsComponent(
                plugin_name="dam-fs",
                asset_storage_path=str(tmp_path),
            ),
            ArchiveSettingsComponent(plugin_name="dam-archive"),
        ],
    )

    # Setup a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    file_hash = await _get_sha256(test_file)

    # Add file to DAM programmatically
    cmd = RegisterLocalFileCommand(file_path=test_file)
    await world.dispatch_command(cmd).get_one_value()

    # Run verification
    results, _, _, _, _ = await verify_assets_logic(
        world=world,
        paths=[test_file],
        recursive=False,
        process_map={},
        stop_on_error=True,
    )

    # Check report
    assert len(results) == 1
    assert results[0]["file_path"] == test_file.name
    assert results[0]["calculated_hash"] == file_hash
    assert results[0]["dam_hash"] == file_hash
    assert results[0]["status"] == "VERIFIED"


@pytest.mark.asyncio
async def test_verify_single_file_fail(
    tmp_path: Path,
    world_factory: WorldFactory,
):
    """Test that a single, modified file fails verification."""
    # Create a test world
    world = await world_factory(
        "test_world",
        [
            FsSettingsComponent(
                plugin_name="dam-fs",
                asset_storage_path=str(tmp_path),
            ),
            ArchiveSettingsComponent(plugin_name="dam-archive"),
        ],
    )

    # Setup a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    # Add file to DAM programmatically
    cmd = RegisterLocalFileCommand(file_path=test_file)
    await world.dispatch_command(cmd).get_one_value()

    # Modify the file
    test_file.write_text("world")
    new_hash = await _get_sha256(test_file)

    # Run verification
    results, _, _, _, _ = await verify_assets_logic(
        world=world,
        paths=[test_file],
        recursive=False,
        process_map={},
        stop_on_error=True,
    )

    # Check report
    assert len(results) == 1
    assert results[0]["status"] == "FAILED"
    assert results[0]["calculated_hash"] == new_hash


@pytest.mark.asyncio
async def test_verify_archive_ok(
    tmp_path: Path,
    world_factory: WorldFactory,
):
    """Test that an unmodified archive and its contents pass verification."""
    # Create a test world
    world = await world_factory(
        "test_world",
        [
            FsSettingsComponent(
                plugin_name="dam-fs",
                asset_storage_path=str(tmp_path),
            ),
            ArchiveSettingsComponent(plugin_name="dam-archive"),
        ],
    )

    # Create a zip file
    zip_path = tmp_path / "test.zip"
    file1 = tmp_path / "file1.txt"
    file1.write_text("one")
    file2 = tmp_path / "file2.txt"
    file2.write_text("two")

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(file1, "file1.txt")
        zf.write(file2, "file2.txt")

    # Add archive to DAM and process it
    register_cmd = RegisterLocalFileCommand(file_path=zip_path)
    entity_id = await world.dispatch_command(register_cmd).get_one_value()
    assert entity_id is not None

    # Set MIME type before ingestion
    set_mime_cmd = AutoSetMimeTypeCommand(entity_id=entity_id)
    await world.dispatch_command(set_mime_cmd).get_all_results()

    ingest_cmd = IngestArchiveCommand(entity_id=entity_id)
    await world.dispatch_command(ingest_cmd).get_all_results()

    # Run verification
    results, _, _, _, _ = await verify_assets_logic(
        world=world,
        paths=[zip_path],
        recursive=False,
        process_map={".zip": ["VerifyArchiveContentsCommand"]},
        stop_on_error=True,
    )

    # Check report
    assert len(results) == 3
    assert results[0]["status"] == "VERIFIED"
    assert results[1]["status"] == "VERIFIED"
    assert results[2]["status"] == "VERIFIED"
