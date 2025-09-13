import asyncio
import pytest
from pytest_mock import MockerFixture
from typing import Annotated

from dam.core.world import World
from dam.core.transaction import EcsTransaction
from sqlalchemy import select

from dam_archive.commands import ExtractArchiveMembersCommand
from dam_archive.models import (
    SplitArchiveManifestComponent,
    SplitArchivePartInfoComponent,
)
from dam_fs.models import FilePropertiesComponent


@pytest.mark.serial
@pytest.mark.asyncio
async def test_split_archive_assembly_from_command(
    test_world_alpha: Annotated[World, "Resource"],
    mocker: MockerFixture,
):
    """
    Tests that the assembly logic in the extraction handler works correctly
    by manually setting up the component state first.
    """
    world = test_world_alpha
    perform_extraction_mock = mocker.patch(
        "dam_archive.systems._perform_extraction", return_value=None
    )

    # Mock the GetAssetStreamCommand to avoid needing real files
    original_dispatch = world.dispatch_command
    async def mock_dispatch(command):
        import io
        from dam.core.commands import CommandResult
        from dam.commands import GetAssetStreamCommand
        from dam.core.result import HandlerResult

        if isinstance(command, GetAssetStreamCommand):
            return CommandResult(results=[HandlerResult(value=io.BytesIO(b"dummy"))])
        return await original_dispatch(command)

    mocker.patch.object(world, 'dispatch_command', side_effect=mock_dispatch)


    entity_id_1, entity_id_2 = 0, 0
    base_name = "test"

    # 1. Setup: Manually create the entities and components
    async with world.db_session_maker() as session:
        transaction = EcsTransaction(session)

        # Part 1
        entity1 = await transaction.create_entity()
        entity_id_1 = entity1.id
        await transaction.add_component_to_entity(entity_id_1, FilePropertiesComponent(original_filename=f"{base_name}.part1.rar"))
        await transaction.add_component_to_entity(entity_id_1, SplitArchivePartInfoComponent(base_name=base_name, part_num=1))

        # Part 2
        entity2 = await transaction.create_entity()
        entity_id_2 = entity2.id
        await transaction.add_component_to_entity(entity_id_2, FilePropertiesComponent(original_filename=f"{base_name}.part2.rar"))
        await transaction.add_component_to_entity(entity_id_2, SplitArchivePartInfoComponent(base_name=base_name, part_num=2))

        await session.commit()

    # 2. Action: Request extraction on the first part
    extract_cmd = ExtractArchiveMembersCommand(entity_id=entity_id_1)
    await world.dispatch_command(extract_cmd)
    await asyncio.sleep(0.1) # Wait for async operations

    # 3. Assertions
    async with world.db_session_maker() as session:
        # A master entity should have been created
        manifest_query = await session.execute(select(SplitArchiveManifestComponent))
        manifests = manifest_query.scalars().all()
        assert len(manifests) == 1
        manifest = manifests[0]
        master_entity_id = manifest.entity_id
        assert manifest.part_entity_ids == [entity_id_1, entity_id_2]

        # Verify the mock was called correctly
        perform_extraction_mock.assert_called_once()
        call_args = perform_extraction_mock.call_args[0]
        assert call_args[0] == master_entity_id
