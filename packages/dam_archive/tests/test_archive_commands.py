"""Tests for archive commands."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.models.metadata.content_length_component import ContentLengthComponent
from dam_fs.commands import RegisterLocalFileCommand
from dam_fs.models.filename_component import FilenameComponent
from dam_fs.settings import FsSettingsComponent
from dam_test_utils.types import WorldFactory
from sqlalchemy import select

from dam_archive.commands.split_archives import (
    BindSplitArchiveCommand,
    CreateMasterArchiveCommand,
    UnbindSplitArchiveCommand,
)
from dam_archive.models import (
    SplitArchiveManifestComponent,
    SplitArchivePartInfoComponent,
)
from dam_archive.settings import ArchiveSettingsComponent


@pytest.fixture
def fs_settings(tmp_path: Path) -> FsSettingsComponent:
    """Create a FsSettingsComponent for testing."""
    asset_storage_path = tmp_path / "asset_storage"
    asset_storage_path.mkdir()
    return FsSettingsComponent(
        plugin_name="dam-fs",
        asset_storage_path=str(asset_storage_path),
    )


@pytest.fixture
def archive_settings() -> ArchiveSettingsComponent:
    """Create an ArchiveSettingsComponent for testing."""
    return ArchiveSettingsComponent(plugin_name="dam-archive")


@pytest.mark.serial
@pytest.mark.asyncio
async def test_bind_split_archive_command_workflow(
    world_factory: WorldFactory,
    fs_settings: FsSettingsComponent,
    archive_settings: ArchiveSettingsComponent,
    tmp_path: Path,
):
    """
    Test that the BindSplitArchiveCommand correctly finds and assembles a split archive.

    This test checks the workflow when the command is triggered from a single part entity.
    """
    world: World = await world_factory("test_world", [fs_settings, archive_settings])
    base_name = "test_bind_command"
    entity_ids: list[int] = []

    # 1. Setup: Register local files to create entities with all necessary components
    for i in range(1, 3):
        part_file = tmp_path / f"{base_name}.part{i}.rar"
        part_file.write_text(f"content of part {i}")
        register_cmd = RegisterLocalFileCommand(file_path=part_file)
        entity_id = await world.dispatch_command(register_cmd).get_one_value()
        assert entity_id is not None
        entity_ids.append(entity_id)

    # 2. Action: Run the binding command on the first part
    bind_cmd = BindSplitArchiveCommand(entity_id=entity_ids[0])
    async for _ in world.dispatch_command(bind_cmd):
        pass
    await asyncio.sleep(0.1)

    # 3. Assertions
    tm = world.get_context(WorldTransaction)
    async with tm() as transaction:
        session = transaction.session
        manifest_query = await session.execute(select(SplitArchiveManifestComponent))
        manifests = manifest_query.scalars().all()
        assert len(manifests) == 1
        master_entity_id = manifests[0].entity_id

        part_info_query = await session.execute(
            select(SplitArchivePartInfoComponent)
            .where(SplitArchivePartInfoComponent.master_entity_id == master_entity_id)
            .order_by(SplitArchivePartInfoComponent.part_num)
        )
        parts = part_info_query.scalars().all()
        assert [part.entity_id for part in parts] == entity_ids


@pytest.mark.serial
@pytest.mark.asyncio
async def test_bind_split_archive_operation_workflow(
    world_factory: WorldFactory,
    fs_settings: FsSettingsComponent,
    archive_settings: ArchiveSettingsComponent,
    tmp_path: Path,
):
    """
    Test the full lifecycle of the 'archive.bind-split-archive' AssetOperation.

    This test covers the add, check, and remove commands of the operation.
    """
    world: World = await world_factory("test_world", [fs_settings, archive_settings])
    base_name = "test_operation"
    entity_ids: list[int] = []
    tm = world.get_context(WorldTransaction)

    # 1. Setup: Register part files as entities
    for i in range(1, 3):
        part_file = tmp_path / f"{base_name}.part{i}.rar"
        part_file.write_text(f"content of part {i}")
        register_cmd = RegisterLocalFileCommand(file_path=part_file)
        entity_id = await world.dispatch_command(register_cmd).get_one_value()
        assert entity_id is not None
        entity_ids.append(entity_id)

    # 3. Action (Add): Bind the archive using the operation's command
    bind_cmd = BindSplitArchiveCommand(entity_id=entity_ids[0])
    async for _ in world.dispatch_command(bind_cmd):
        pass
    await asyncio.sleep(0.1)

    # 4. Assertion (Check): Verify binding is complete
    async with tm() as transaction:
        manifest_query = await transaction.session.execute(select(SplitArchiveManifestComponent))
        manifests = manifest_query.scalars().all()
        assert len(manifests) == 1


@pytest.mark.serial
@pytest.mark.asyncio
async def test_manual_create_and_unbind_workflow(
    world_factory: WorldFactory,
    fs_settings: FsSettingsComponent,
    archive_settings: ArchiveSettingsComponent,
):
    """Test that manual creation and unbinding of a master archive works."""
    world: World = await world_factory("test_world", [fs_settings, archive_settings])
    entity_ids: list[int] = []

    # 1. Setup: Manually create part entities
    tm = world.get_context(WorldTransaction)
    async with tm() as transaction:
        for i in range(1, 3):
            entity = await transaction.create_entity()
            entity_ids.append(entity.id)
            await transaction.add_component_to_entity(
                entity.id, FilenameComponent(filename=f"manual.part{i}.rar", first_seen_at=datetime.now(UTC))
            )
            await transaction.add_component_to_entity(entity.id, ContentLengthComponent(file_size_bytes=100))
        await transaction.session.commit()

    # 2. Action: Create the master entity manually
    create_cmd = CreateMasterArchiveCommand(name="manual_master", part_entity_ids=entity_ids)
    async for _ in world.dispatch_command(create_cmd):
        pass
    await asyncio.sleep(0.1)

    # 3. Assertions for creation
    master_entity_id = -1
    async with tm() as transaction:
        session = transaction.session
        manifest_query = await session.execute(select(SplitArchiveManifestComponent))
        manifests = manifest_query.scalars().all()
        assert len(manifests) == 1
        master_entity_id = manifests[0].entity_id

        part_info_query = await session.execute(
            select(SplitArchivePartInfoComponent)
            .where(SplitArchivePartInfoComponent.master_entity_id == master_entity_id)
            .order_by(SplitArchivePartInfoComponent.part_num)
        )
        parts = part_info_query.scalars().all()
        assert [part.entity_id for part in parts] == entity_ids

    # 4. Action: Unbind the archive from the master
    unbind_cmd = UnbindSplitArchiveCommand(entity_id=master_entity_id)
    async for _ in world.dispatch_command(unbind_cmd):
        pass
    await asyncio.sleep(0.1)

    # 5. Assertions for unbinding
    async with tm() as transaction:
        session = transaction.session
        manifest_query = await session.execute(select(SplitArchiveManifestComponent))
        assert len(manifest_query.scalars().all()) == 0
        part_info_query = await session.execute(select(SplitArchivePartInfoComponent))
        assert len(part_info_query.scalars().all()) == 0
