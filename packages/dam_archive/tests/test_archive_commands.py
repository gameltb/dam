import asyncio
from pathlib import Path
from typing import Annotated, List

import pytest
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam_fs.commands import RegisterLocalFileCommand
from sqlalchemy import select

from dam_archive.commands import (
    BindSplitArchiveCommand,
    CreateMasterArchiveCommand,
    UnbindSplitArchiveCommand,
)
from dam_archive.models import (
    SplitArchiveManifestComponent,
    SplitArchivePartInfoComponent,
)


@pytest.mark.serial
@pytest.mark.asyncio
async def test_bind_split_archive_command_workflow(
    test_world_alpha: Annotated[World, "Resource"],
    tmp_path: Path,
):
    """
    Tests that the BindSplitArchiveCommand correctly finds and assembles a split archive
    when triggered from a single part entity.
    """
    world = test_world_alpha
    base_name = "test_bind_command"
    entity_ids: List[int] = []

    # 1. Setup: Register local files to create entities with all necessary components
    for i in range(1, 3):
        part_file = tmp_path / f"{base_name}.part{i}.rar"
        part_file.write_text(f"content of part {i}")
        register_cmd = RegisterLocalFileCommand(file_path=part_file)
        entity_id = await world.dispatch_command(register_cmd).get_one_value()
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
    test_world_alpha: Annotated[World, "Resource"],
    tmp_path: Path,
):
    """
    Tests the full lifecycle of the 'archive.bind-split-archive' AssetOperation:
    add, check, and remove.
    """
    world = test_world_alpha
    base_name = "test_operation"
    entity_ids: List[int] = []
    tm = world.get_context(WorldTransaction)

    # 1. Setup: Register part files as entities
    for i in range(1, 3):
        part_file = tmp_path / f"{base_name}.part{i}.rar"
        part_file.write_text(f"content of part {i}")
        register_cmd = RegisterLocalFileCommand(file_path=part_file)
        entity_id = await world.dispatch_command(register_cmd).get_one_value()
        entity_ids.append(entity_id)

    # 2. Get the AssetOperation
    operation = world.get_asset_operation("archive.bind-split-archive")
    assert operation is not None
    assert operation.add_command_class is not None
    assert operation.check_command_class is not None
    assert operation.remove_command_class is not None

    # 3. Action (Add): Bind the archive using the operation's command
    bind_cmd = operation.add_command_class(entity_id=entity_ids[0])
    async for _ in world.dispatch_command(bind_cmd):
        pass
    await asyncio.sleep(0.1)

    # 4. Assertion (Check): Verify binding is complete
    master_entity_id = -1
    async with tm() as transaction:
        manifest_query = await transaction.session.execute(select(SplitArchiveManifestComponent))
        manifests = manifest_query.scalars().all()
        assert len(manifests) == 1
        master_entity_id = manifests[0].entity_id

    # Check from a part
    check_cmd_part = operation.check_command_class(entity_id=entity_ids[1])
    is_bound_part = await world.dispatch_command(check_cmd_part).get_one_value()
    assert is_bound_part is True

    # Check from the master
    check_cmd_master = operation.check_command_class(entity_id=master_entity_id)
    is_bound_master = await world.dispatch_command(check_cmd_master).get_one_value()
    assert is_bound_master is True

    # 5. Action (Remove): Unbind the archive starting from a part
    unbind_cmd = operation.remove_command_class(entity_id=entity_ids[0])
    async for _ in world.dispatch_command(unbind_cmd):
        pass
    await asyncio.sleep(0.1)

    # 6. Assertion (Final Check): Verify it's unbound
    is_bound_part_after = await world.dispatch_command(check_cmd_part).get_one_value()
    assert is_bound_part_after is False

    is_bound_master_after = await world.dispatch_command(check_cmd_master).get_one_value()
    assert is_bound_master_after is False

    async with tm() as transaction:
        manifest_query = await transaction.session.execute(select(SplitArchiveManifestComponent))
        assert len(manifest_query.scalars().all()) == 0
        part_info_query = await transaction.session.execute(select(SplitArchivePartInfoComponent))
        assert len(part_info_query.scalars().all()) == 0


@pytest.mark.serial
@pytest.mark.asyncio
async def test_manual_create_and_unbind_workflow(
    test_world_alpha: Annotated[World, "Resource"],
):
    """
    Tests that manual creation and unbinding of a master archive works.
    """
    world = test_world_alpha
    entity_ids: List[int] = []

    # 1. Setup: Manually create part entities
    tm = world.get_context(WorldTransaction)
    async with tm() as transaction:
        from datetime import datetime, timezone

        from dam.models.metadata.content_length_component import ContentLengthComponent
        from dam_fs.models import FilenameComponent

        for i in range(1, 3):
            entity = await transaction.create_entity()
            entity_ids.append(entity.id)
            await transaction.add_component_to_entity(
                entity.id, FilenameComponent(filename=f"manual.part{i}.rar", first_seen_at=datetime.now(timezone.utc))
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
