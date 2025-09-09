from pathlib import Path

import pytest
from dam.core.world import World
from dam.functions import ecs_functions
from dam_fs.commands import RegisterLocalFileCommand
from sqlalchemy import select

from dam_archive.commands import ExtractArchiveMembersCommand
from dam_archive.models import ArchiveInfoComponent, ArchiveMemberComponent


@pytest.mark.serial
@pytest.mark.asyncio
async def test_extract_archives(test_world_alpha: World, test_archives: tuple[Path, Path]) -> None:
    """
    Tests extracting both regular and protected archives using the new fixture.
    """
    world = test_world_alpha
    regular_archive_path, protected_archive_path = test_archives

    # --- Test Regular Archive ---
    # 1. Register the regular archive file
    register_cmd_reg = RegisterLocalFileCommand(file_path=regular_archive_path)
    cmd_result_reg = await world.dispatch_command(register_cmd_reg)
    entity_id_reg = cmd_result_reg.get_one_value()

    # 2. Run the extraction command
    extract_cmd_reg = ExtractArchiveMembersCommand(entity_id=entity_id_reg)
    await world.dispatch_command(extract_cmd_reg)

    # 3. Verify the results for the regular archive
    async with world.db_session_maker() as session:
        info_reg = await ecs_functions.get_component(session, entity_id_reg, ArchiveInfoComponent)
        assert info_reg is not None
        assert info_reg.file_count == 2

        members_reg = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id_reg)
        )
        assert len(members_reg.scalars().all()) == 2

    # --- Test Protected Archive ---
    # 1. Register the protected archive file
    register_cmd_prot = RegisterLocalFileCommand(file_path=protected_archive_path)
    cmd_result_prot = await world.dispatch_command(register_cmd_prot)
    entity_id_prot = cmd_result_prot.get_one_value()

    # 2. Run the extraction command with the correct password
    extract_cmd_prot = ExtractArchiveMembersCommand(entity_id=entity_id_prot, passwords=["password"])
    await world.dispatch_command(extract_cmd_prot)

    # 3. Verify the results for the protected archive
    async with world.db_session_maker() as session:
        info_prot = await ecs_functions.get_component(session, entity_id_prot, ArchiveInfoComponent)
        assert info_prot is not None
        assert info_prot.file_count == 2

        members_prot = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id_prot)
        )
        assert len(members_prot.scalars().all()) == 2
