import datetime
import shutil
from pathlib import Path

import pytest
from dam.core.world import World
from dam.functions import ecs_functions
from dam_source.models.source_info import OriginalSourceInfoComponent

from dam_fs.commands import FindEntityByFilePropertiesCommand, RegisterLocalFileCommand
from dam_fs.models import FileLocationComponent, FilePropertiesComponent


@pytest.mark.serial
@pytest.mark.asyncio
async def test_register_and_find(test_world_alpha: World, temp_asset_file: Path):
    """
    Tests the full flow of registering a new file and then finding it by its properties.
    """
    world = test_world_alpha

    # 1. Register a new file
    register_cmd = RegisterLocalFileCommand(file_path=temp_asset_file)
    cmd_result = await world.dispatch_command(register_cmd)
    entity_id = cmd_result.get_one_value()

    assert isinstance(entity_id, int)

    # Verify components for the new entity
    async with world.db_session_maker() as session:
        fpc = await ecs_functions.get_component(session, entity_id, FilePropertiesComponent)
        assert fpc is not None
        assert fpc.original_filename == temp_asset_file.name
        assert fpc.file_size_bytes == temp_asset_file.stat().st_size

        flcs = await ecs_functions.get_components(session, entity_id, FileLocationComponent)
        assert len(flcs) == 1
        assert flcs[0].url == temp_asset_file.as_uri()

        osis = await ecs_functions.get_components(session, entity_id, OriginalSourceInfoComponent)
        assert len(osis) == 1

    # 2. Find the entity by its properties
    mod_time = datetime.datetime.fromtimestamp(temp_asset_file.stat().st_mtime, tz=datetime.timezone.utc)
    find_cmd = FindEntityByFilePropertiesCommand(
        file_path=temp_asset_file.as_uri(),
        file_modified_at=mod_time,
    )
    find_result = await world.dispatch_command(find_cmd)
    found_entity_id = find_result.get_one_value()

    assert found_entity_id == entity_id

    # 3. Register a duplicate file (same hash, different path)
    duplicate_dir = temp_asset_file.parent / "duplicate"
    duplicate_dir.mkdir()
    duplicate_file = duplicate_dir / temp_asset_file.name
    shutil.copy(temp_asset_file, duplicate_file)

    register_dup_cmd = RegisterLocalFileCommand(file_path=duplicate_file)
    dup_cmd_result = await world.dispatch_command(register_dup_cmd)
    dup_entity_id = dup_cmd_result.get_one_value()

    # Should be the same entity
    assert dup_entity_id == entity_id

    # Verify there are now two FileLocationComponents
    async with world.db_session_maker() as session:
        flcs_after_dup = await ecs_functions.get_components(session, entity_id, FileLocationComponent)
        assert len(flcs_after_dup) == 2
        urls = {flc.url for flc in flcs_after_dup}
        assert temp_asset_file.as_uri() in urls
        assert duplicate_file.as_uri() in urls


@pytest.mark.serial
@pytest.mark.asyncio
async def test_store_asset(test_world_alpha: World, temp_asset_file: Path):
    """
    Tests the store_assets_handler system.
    """
    world = test_world_alpha

    # 1. Register a local file first
    register_cmd = RegisterLocalFileCommand(file_path=temp_asset_file)
    cmd_result = await world.dispatch_command(register_cmd)
    entity_id = cmd_result.get_one_value()

    # 2. Dispatch the store command
    from dam_fs.commands import StoreAssetsCommand

    store_cmd = StoreAssetsCommand(query="local_not_stored")
    await world.dispatch_command(store_cmd)

    # 3. Verify the new CAS location
    async with world.db_session_maker() as session:
        flcs = await ecs_functions.get_components(session, entity_id, FileLocationComponent)
        assert len(flcs) == 2  # Should now have the original file:/// and the new cas:///

        cas_locs = [flc for flc in flcs if flc.url.startswith("cas://")]
        assert len(cas_locs) == 1

        # Check that the stored file exists
        from dam_fs.resources import FileStorageResource

        storage = world.get_resource(FileStorageResource)
        cas_hash = cas_locs[0].url.split("://")[1]
        stored_path = storage.get_file_path(cas_hash)
        assert stored_path is not None
        assert stored_path.exists()
