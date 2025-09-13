import asyncio
from pathlib import Path
from typing import Annotated, List

import pytest
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam_fs.models import FilePropertiesComponent
from sqlalchemy import select

from dam_archive.commands import (
    CreateMasterArchiveCommand,
    DiscoverAndBindCommand,
    UnbindSplitArchiveCommand,
)
from dam_archive.models import (
    SplitArchiveManifestComponent,
    SplitArchivePartInfoComponent,
)


@pytest.mark.serial
@pytest.mark.asyncio
async def test_discover_and_bind_workflow(
    test_world_alpha: Annotated[World, "Resource"],
    tmp_path: Path,
):
    """
    Tests that the DiscoverAndBindCommand correctly finds, tags, and assembles a split archive.
    """
    world = test_world_alpha
    base_name = "test_discover"
    entity_ids: List[int] = []

    # 1. Setup: Manually create entities, but without the SplitArchivePartInfoComponent
    async with world.db_session_maker() as session:
        transaction = EcsTransaction(session)
        for i in range(1, 3):
            part_file = tmp_path / f"{base_name}.part{i}.rar"
            part_file.touch()
            entity = await transaction.create_entity()
            entity_ids.append(entity.id)
            await transaction.add_component_to_entity(
                entity.id, FilePropertiesComponent(original_filename=str(part_file))
            )
        await session.commit()

    # 2. Action: Run the discovery and binding command
    discover_cmd = DiscoverAndBindCommand(paths=[str(tmp_path)])
    await world.dispatch_command(discover_cmd)
    await asyncio.sleep(0.1)

    # 3. Assertions
    async with world.db_session_maker() as session:
        manifest_query = await session.execute(select(SplitArchiveManifestComponent))
        manifests = manifest_query.scalars().all()
        assert len(manifests) == 1
        assert manifests[0].part_entity_ids == entity_ids

        for entity_id in entity_ids:
            part_info = await session.get(SplitArchivePartInfoComponent, entity_id)
            assert part_info is not None
            assert part_info.master_entity_id == manifests[0].entity_id


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
    async with world.db_session_maker() as session:
        transaction = EcsTransaction(session)
        for i in range(1, 3):
            entity = await transaction.create_entity()
            entity_ids.append(entity.id)
            await transaction.add_component_to_entity(
                entity.id, FilePropertiesComponent(original_filename=f"manual.part{i}.rar")
            )
        await session.commit()

    # 2. Action: Create the master entity manually
    create_cmd = CreateMasterArchiveCommand(name="manual_master", part_entity_ids=entity_ids)
    await world.dispatch_command(create_cmd)
    await asyncio.sleep(0.1)

    # 3. Assertions for creation
    master_entity_id = -1
    async with world.db_session_maker() as session:
        manifest_query = await session.execute(select(SplitArchiveManifestComponent))
        manifests = manifest_query.scalars().all()
        assert len(manifests) == 1
        master_entity_id = manifests[0].entity_id
        assert manifests[0].part_entity_ids == entity_ids

    # 4. Action: Unbind the archive
    unbind_cmd = UnbindSplitArchiveCommand(master_entity_id=master_entity_id)
    await world.dispatch_command(unbind_cmd)
    await asyncio.sleep(0.1)

    # 5. Assertions for unbinding
    async with world.db_session_maker() as session:
        manifest_query = await session.execute(select(SplitArchiveManifestComponent))
        assert len(manifest_query.scalars().all()) == 0
        part_info_query = await session.execute(select(SplitArchivePartInfoComponent))
        assert len(part_info_query.scalars().all()) == 0
