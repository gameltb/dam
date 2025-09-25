from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

import pytest
from dam.commands.analysis_commands import AutoSetMimeTypeCommand
from dam.core.world import World
from dam_archive.commands import IngestArchiveCommand
from dam_fs.commands import RegisterLocalFileCommand

from dam_app.cli.verify import verify_assets
from dam_app.state import global_state


def _get_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


@pytest.mark.asyncio
async def test_verify_single_file_ok(
    tmp_path: Path,
    test_world_alpha: World,
):
    # Setup a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    file_hash = _get_sha256(test_file)

    # Add file to DAM programmatically
    cmd = RegisterLocalFileCommand(file_path=test_file)
    await test_world_alpha.dispatch_command(cmd).get_one_value()

    # Set the world for the command to use
    global_state.world_name = test_world_alpha.name

    # Run verification
    await verify_assets(paths=[test_file], recursive=False, process=None, stop_on_error=True)

    # Check report
    report_files = list(Path.cwd().glob("verification_report_*.csv"))
    try:
        assert len(report_files) == 1
        with open(report_files[0], "r") as f:
            reader = csv.DictReader(f)
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
):
    # Setup a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    # Add file to DAM programmatically
    cmd = RegisterLocalFileCommand(file_path=test_file)
    await test_world_alpha.dispatch_command(cmd).get_one_value()

    # Modify the file
    test_file.write_text("world")
    new_hash = _get_sha256(test_file)

    # Set the world for the command to use
    global_state.world_name = test_world_alpha.name

    # Run verification
    await verify_assets(paths=[test_file], recursive=False, process=None, stop_on_error=True)

    # Check report
    report_files = list(Path.cwd().glob("verification_report_*.csv"))
    try:
        assert len(report_files) == 1
        with open(report_files[0], "r") as f:
            reader = csv.DictReader(f)
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
):
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

    # Set MIME type before ingestion
    set_mime_cmd = AutoSetMimeTypeCommand(entity_id=entity_id)
    await test_world_alpha.dispatch_command(set_mime_cmd).get_all_results()

    ingest_cmd = IngestArchiveCommand(entity_id=entity_id)
    await test_world_alpha.dispatch_command(ingest_cmd).get_all_results()

    # Set the world for the command to use
    global_state.world_name = test_world_alpha.name

    # Run verification
    await verify_assets(paths=[zip_path], recursive=False, process=["VerifyArchiveContentsCommand"], stop_on_error=True)

    # Check report
    report_files = list(Path.cwd().glob("verification_report_*.csv"))
    try:
        assert len(report_files) == 1
        with open(report_files[0], "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3
            assert rows[0]["status"] == "VERIFIED"
            assert rows[1]["status"] == "VERIFIED"
            assert rows[2]["status"] == "VERIFIED"
    finally:
        for f in report_files:
            f.unlink()
