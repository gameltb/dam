"""Tests for the discovery system in the `dam_fs` package."""

from pathlib import Path

import pytest
from dam.commands.discovery_commands import DiscoverPathSiblingsCommand
from dam.core.world import World
from dam_test_utils.types import WorldFactory

from dam_fs.commands import RegisterLocalFileCommand
from dam_fs.settings import FsSettingsComponent


@pytest.fixture
def fs_settings(tmp_path: Path) -> FsSettingsComponent:
    """Create a FsSettingsComponent for testing."""
    asset_storage_path = tmp_path / "asset_storage"
    asset_storage_path.mkdir()
    return FsSettingsComponent(
        plugin_name="dam-fs",
        asset_storage_path=str(asset_storage_path),
    )


@pytest.mark.asyncio
async def test_discover_fs_path_siblings(
    world_factory: WorldFactory,
    fs_settings: FsSettingsComponent,
    tmp_path: Path,
):
    """
    Test that the discover_fs_path_siblings_handler finds entities.

    It should correctly find entities in the same filesystem directory and
    return them as PathSibling objects.
    """
    world: World = await world_factory("test_world", [fs_settings])
    entity_ids: list[int] = []

    # 1. Setup: Register two files in the same directory
    file_1 = tmp_path / "file_1.txt"
    file_1.write_text("content 1")
    entity_id_1 = await world.dispatch_command(RegisterLocalFileCommand(file_path=file_1)).get_one_value()
    assert entity_id_1 is not None
    entity_ids.append(entity_id_1)

    file_2 = tmp_path / "file_2.txt"
    file_2.write_text("content 2")
    entity_id_2 = await world.dispatch_command(RegisterLocalFileCommand(file_path=file_2)).get_one_value()
    assert entity_id_2 is not None
    entity_ids.append(entity_id_2)

    # Create a file in a subdirectory to ensure it's NOT found
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    other_file = sub_dir / "other.txt"
    other_file.write_text("other content")
    other_entity_id = await world.dispatch_command(RegisterLocalFileCommand(file_path=other_file)).get_one_value()
    assert other_entity_id is not None

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
