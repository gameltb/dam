from pathlib import Path
from typing import Annotated, List

import pytest
from dam.commands.discovery_commands import DiscoverPathSiblingsCommand
from dam.core.world import World

from dam_fs.commands import RegisterLocalFileCommand


@pytest.mark.asyncio
async def test_discover_fs_path_siblings(
    test_world_alpha: Annotated[World, "Resource"],
    tmp_path: Path,
):
    """
    Tests that the discover_fs_path_siblings_handler correctly finds entities
    in the same filesystem directory and returns them as PathSibling objects.
    """
    world = test_world_alpha
    entity_ids: List[int] = []

    # 1. Setup: Register two files in the same directory
    file_1 = tmp_path / "file_1.txt"
    file_1.write_text("content 1")
    entity_id_1 = await world.dispatch_command(RegisterLocalFileCommand(file_path=file_1)).get_one_value()
    entity_ids.append(entity_id_1)

    file_2 = tmp_path / "file_2.txt"
    file_2.write_text("content 2")
    entity_id_2 = await world.dispatch_command(RegisterLocalFileCommand(file_path=file_2)).get_one_value()
    entity_ids.append(entity_id_2)

    # Create a file in a subdirectory to ensure it's NOT found
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    other_file = sub_dir / "other.txt"
    other_file.write_text("other content")
    other_entity_id = await world.dispatch_command(RegisterLocalFileCommand(file_path=other_file)).get_one_value()

    # 2. Action: Run discovery on the first entity
    discover_cmd = DiscoverPathSiblingsCommand(entity_id=entity_id_1)
    try:
        # Use get_one_value() as it correctly handles the test runner's lifecycle
        discovered_siblings = await world.dispatch_command(discover_cmd).get_one_value()
    except ValueError:
        discovered_siblings = None

    # 3. Assertions
    assert discovered_siblings is not None
    assert len(discovered_siblings) == 2

    # Check that the returned objects have the correct structure and data
    discovered_map = {sib.entity_id: sib.path for sib in discovered_siblings}
    assert set(discovered_map.keys()) == set(entity_ids)
    assert other_entity_id not in discovered_map

    assert discovered_map[entity_id_1] == str(file_1.resolve())
    assert discovered_map[entity_id_2] == str(file_2.resolve())
