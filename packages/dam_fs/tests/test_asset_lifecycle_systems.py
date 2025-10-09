"""Tests for the asset lifecycle systems in the `dam_fs` package."""

import asyncio
import datetime
import os
import shutil
import time
from pathlib import Path

import pytest
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.functions import ecs_functions
from dam.models.hashes import ContentHashSHA256Component
from dam.models.metadata.content_length_component import ContentLengthComponent

from dam_fs.commands import (
    FindEntityByFilePropertiesCommand,
    RegisterLocalFileCommand,
    StoreAssetsCommand,
)
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.models.filename_component import FilenameComponent
from dam_fs.resources.file_storage_resource import FileStorageResource


@pytest.mark.serial
@pytest.mark.asyncio
async def test_register_and_find(test_world_alpha: World, temp_asset_file: Path):
    """Tests the full flow of registering a new file and then finding it by its properties."""
    world = test_world_alpha
    mod_time = datetime.datetime.fromtimestamp(temp_asset_file.stat().st_mtime, tz=datetime.UTC)

    # 1. Register a new file
    register_cmd = RegisterLocalFileCommand(file_path=temp_asset_file)
    entity_id = await world.dispatch_command(register_cmd).get_one_value()

    assert isinstance(entity_id, int)

    # Verify components for the new entity
    async with world.get_context(WorldTransaction)() as tx:
        session = tx.session
        fnc = await ecs_functions.get_component(session, entity_id, FilenameComponent)
        assert fnc is not None
        assert fnc.filename == temp_asset_file.name
        assert fnc.first_seen_at == mod_time

        clc = await ecs_functions.get_component(session, entity_id, ContentLengthComponent)
        assert clc is not None
        assert clc.file_size_bytes == temp_asset_file.stat().st_size

        flc = await ecs_functions.get_component(session, entity_id, FileLocationComponent)
        assert flc is not None
        assert flc.url == temp_asset_file.as_uri()
        assert flc.last_modified_at == mod_time

    # 2. Find the entity by its properties
    find_cmd = FindEntityByFilePropertiesCommand(
        file_path=temp_asset_file.as_uri(),
        last_modified_at=mod_time,
    )
    found_entity_id = await world.dispatch_command(find_cmd).get_one_value()

    assert found_entity_id == entity_id

    # 3. Register a duplicate file (same hash, different path)
    duplicate_dir = temp_asset_file.parent / "duplicate"
    duplicate_dir.mkdir()
    duplicate_file = duplicate_dir / temp_asset_file.name
    shutil.copy(temp_asset_file, duplicate_file)

    register_dup_cmd = RegisterLocalFileCommand(file_path=duplicate_file)
    dup_entity_id = await world.dispatch_command(register_dup_cmd).get_one_value()

    # Should be the same entity
    assert dup_entity_id == entity_id

    # Verify there are now two FileLocationComponents
    async with world.get_context(WorldTransaction)() as tx:
        session = tx.session
        flcs_after_dup = await ecs_functions.get_components(session, entity_id, FileLocationComponent)
        assert len(flcs_after_dup) == 2
        urls = {flc.url for flc in flcs_after_dup}
        assert temp_asset_file.as_uri() in urls
        assert duplicate_file.as_uri() in urls


@pytest.mark.serial
@pytest.mark.asyncio
async def test_first_seen_at_logic(test_world_alpha: World, tmp_path: Path):
    """
    Test that the first_seen_at timestamp is correctly updated.

    It should be updated to the earliest known time.
    """
    world = test_world_alpha

    # 1. Create and register a file with a recent timestamp
    recent_file = tmp_path / "test_file.txt"
    recent_file.write_text("same content")
    recent_file.touch()  # Update mtime to now
    recent_mod_time = datetime.datetime.fromtimestamp(recent_file.stat().st_mtime, tz=datetime.UTC)

    register_cmd1 = RegisterLocalFileCommand(file_path=recent_file)
    entity_id = await world.dispatch_command(register_cmd1).get_one_value()

    async with world.get_context(WorldTransaction)() as tx:
        session = tx.session
        fnc = await ecs_functions.get_component(session, entity_id, FilenameComponent)
        assert fnc is not None
        assert fnc.first_seen_at is not None
        assert fnc.first_seen_at == recent_mod_time

    # 2. Create another file with the same name and content but an earlier timestamp
    await asyncio.sleep(1)
    earlier_file_dir = tmp_path / "earlier"
    earlier_file_dir.mkdir()
    earlier_file = earlier_file_dir / "test_file.txt"
    earlier_file.write_text("same content")
    earlier_mtime_val = time.time() - 1000  # 1000 seconds in the past

    # Manually set older modification time
    os.utime(earlier_file, (earlier_mtime_val, earlier_mtime_val))
    earlier_mod_time = datetime.datetime.fromtimestamp(earlier_mtime_val, tz=datetime.UTC)

    # 3. Register the older file
    register_cmd2 = RegisterLocalFileCommand(file_path=earlier_file)
    entity_id2 = await world.dispatch_command(register_cmd2).get_one_value()

    assert entity_id2 == entity_id  # Should be the same entity due to content hash

    # 4. Verify that first_seen_at has been updated to the earlier time
    async with world.get_context(WorldTransaction)() as tx:
        session = tx.session
        fnc_after = await ecs_functions.get_component(session, entity_id, FilenameComponent)
        assert fnc_after is not None
        assert fnc_after.first_seen_at == earlier_mod_time


@pytest.mark.serial
@pytest.mark.asyncio
async def test_reregister_modified_file(test_world_alpha: World, temp_asset_file: Path):
    """
    Test that re-registering a modified file updates its timestamp.

    The last_modified_at timestamp should be updated.
    """
    world = test_world_alpha

    # 1. Register the file for the first time
    register_cmd = RegisterLocalFileCommand(file_path=temp_asset_file)
    entity_id = await world.dispatch_command(register_cmd).get_one_value()

    async with world.get_context(WorldTransaction)() as tx:
        session = tx.session
        flc = await ecs_functions.get_component(session, entity_id, FileLocationComponent)
        assert flc is not None
        assert flc.last_modified_at is not None
        original_mtime = flc.last_modified_at

    # 2. Modify the file
    await asyncio.sleep(1)  # Ensure mtime changes
    temp_asset_file.touch()
    new_mod_time = datetime.datetime.fromtimestamp(temp_asset_file.stat().st_mtime, tz=datetime.UTC)

    assert new_mod_time > original_mtime

    # 3. Register the same file again
    register_again_cmd = RegisterLocalFileCommand(file_path=temp_asset_file)
    await world.dispatch_command(register_again_cmd).get_all_results()

    # 4. Verify the timestamp has been updated
    async with world.get_context(WorldTransaction)() as tx:
        session = tx.session
        flc_after = await ecs_functions.get_component(session, entity_id, FileLocationComponent)
        assert flc_after is not None
        assert flc_after.last_modified_at == new_mod_time


@pytest.mark.serial
@pytest.mark.asyncio
async def test_store_asset(test_world_alpha: World, temp_asset_file: Path):
    """Tests the store_assets_handler system."""
    world = test_world_alpha

    # 1. Register a local file first
    register_cmd = RegisterLocalFileCommand(file_path=temp_asset_file)
    entity_id = await world.dispatch_command(register_cmd).get_one_value()

    # 2. Dispatch the store command
    store_cmd = StoreAssetsCommand(query="local_not_stored")
    await world.dispatch_command(store_cmd).get_all_results()

    # 3. Verify the file is in storage
    async with world.get_context(WorldTransaction)() as tx:
        session = tx.session
        sha256_comp = await ecs_functions.get_component(session, entity_id, ContentHashSHA256Component)
        assert sha256_comp is not None

        storage_resource = world.get_resource(FileStorageResource)
        content_hash = sha256_comp.hash_value.hex()
        assert storage_resource.has_file(content_hash)
