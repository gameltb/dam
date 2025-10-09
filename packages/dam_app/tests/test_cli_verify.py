"""Tests for the asset verification CLI command."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import zipfile
from pathlib import Path

import aiofiles
import pytest
from dam.commands.analysis_commands import AutoSetMimeTypeCommand
from dam.core.world import World
from dam_archive.commands import IngestArchiveCommand
from dam_fs.commands import RegisterLocalFileCommand
from pytest_mock import MockerFixture

from dam_app.cli.verify import verify_assets


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
    test_world_alpha: World,
    mocker: MockerFixture,
):
    """Test that a single, unmodified file passes verification."""
    # Patch the get_world function to return our test world
    mocker.patch("dam_app.cli.verify.get_world", return_value=test_world_alpha)

    # Setup a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    file_hash = await _get_sha256(test_file)

    # Add file to DAM programmatically
    cmd = RegisterLocalFileCommand(file_path=test_file)
    await test_world_alpha.dispatch_command(cmd).get_one_value()

    # Run verification
    os.chdir(tmp_path)
    await verify_assets(paths=[test_file], recursive=False, process=None, stop_on_error=True)

    # Check report
    report_files = list(tmp_path.glob("verification_report_*.csv"))
    try:
        assert len(report_files) == 1
        async with aiofiles.open(report_files[0]) as f:
            content = await f.read()
        string_io = io.StringIO(content)
        reader = csv.DictReader(string_io)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["file_path"] == test_file.name
        assert rows[0]["calculated_hash"] == file_hash
        assert rows[0]["dam_hash"] == file_hash
        assert rows[0]["status"] == "VERIFIED"
    finally:
        for f in report_files:
            f.unlink()


@pytest.mark.asyncio
async def test_verify_single_file_fail(
    tmp_path: Path,
    test_world_alpha: World,
    mocker: MockerFixture,
):
    """Test that a single, modified file fails verification."""
    # Patch the get_world function to return our test world
    mocker.patch("dam_app.cli.verify.get_world", return_value=test_world_alpha)

    # Setup a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    # Add file to DAM programmatically
    cmd = RegisterLocalFileCommand(file_path=test_file)
    await test_world_alpha.dispatch_command(cmd).get_one_value()

    # Modify the file
    test_file.write_text("world")
    new_hash = await _get_sha256(test_file)

    # Run verification
    os.chdir(tmp_path)
    await verify_assets(paths=[test_file], recursive=False, process=None, stop_on_error=True)

    # Check report
    report_files = list(tmp_path.glob("verification_report_*.csv"))
    try:
        assert len(report_files) == 1
        async with aiofiles.open(report_files[0]) as f:
            content = await f.read()
        string_io = io.StringIO(content)
        reader = csv.DictReader(string_io)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["status"] == "FAILED"
        assert rows[0]["calculated_hash"] == new_hash
    finally:
        for f in report_files:
            f.unlink()


@pytest.mark.asyncio
async def test_verify_archive_ok(
    tmp_path: Path,
    test_world_alpha: World,
    mocker: MockerFixture,
):
    """Test that an unmodified archive and its contents pass verification."""
    # Patch the get_world function to return our test world
    mocker.patch("dam_app.cli.verify.get_world", return_value=test_world_alpha)

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
    entity_id = await test_world_alpha.dispatch_command(register_cmd).get_one_value()
    assert entity_id is not None

    # Set MIME type before ingestion
    set_mime_cmd = AutoSetMimeTypeCommand(entity_id=entity_id)
    await test_world_alpha.dispatch_command(set_mime_cmd).get_all_results()

    ingest_cmd = IngestArchiveCommand(entity_id=entity_id)
    await test_world_alpha.dispatch_command(ingest_cmd).get_all_results()

    # Run verification
    os.chdir(tmp_path)
    await verify_assets(
        paths=[zip_path], recursive=False, process=[".zip:VerifyArchiveContentsCommand"], stop_on_error=True
    )

    # Check report
    report_files = list(tmp_path.glob("verification_report_*.csv"))
    try:
        assert len(report_files) == 1
        async with aiofiles.open(report_files[0]) as f:
            content = await f.read()
        string_io = io.StringIO(content)
        reader = csv.DictReader(string_io)
        rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["status"] == "VERIFIED"
        assert rows[1]["status"] == "VERIFIED"
        assert rows[2]["status"] == "VERIFIED"
    finally:
        for f in report_files:
            f.unlink()