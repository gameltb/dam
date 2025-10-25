import hashlib
import zipfile
from pathlib import Path

import aiofiles
import pytest
from dam import world_manager
from dam_test_utils.types import WorldFactory

from dam_app.cli.report import execute_delete_report
from dam_app.state import global_state


@pytest.mark.asyncio
async def test_execute_delete_report_partial_archive(world_factory: WorldFactory, tmp_path: Path):
    """Test that execute_delete_report correctly handles partially duplicate archives."""
    world = await world_factory("test_world", [])
    world_manager.register_world(world)
    global_state.world_name = "test_world"

    # Create a dummy archive with one duplicate and one unique file
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("duplicate.txt", b"duplicate content")
        zf.writestr("unique.txt", b"unique content")

    # Create the delete plan CSV
    csv_path = tmp_path / "delete_plan.csv"
    async with aiofiles.open(csv_path, "w", newline="") as csvfile:
        await csvfile.write(
            "source_path,target_path,hash,size,details\n"
            f"/some/source/path,{archive_path},{hashlib.sha256(b'archive content').hexdigest()},123,\"Duplicate members size (123 bytes) exceeds threshold: ['/path/to/archive.zip -> duplicate.txt' is a duplicate of '/some/source/path']\"\n"
        )

    await execute_delete_report(csv_path=csv_path)

    # Verify the outcome
    assert not archive_path.exists()
    extract_dir = tmp_path / "archive"
    assert extract_dir.exists()
    assert (extract_dir / "unique.txt").exists()
    assert not (extract_dir / "duplicate.txt").exists()
    async with aiofiles.open(extract_dir / "unique.txt", "rb") as f:
        assert await f.read() == b"unique content"

    world_manager.unregister_world("test_world")
