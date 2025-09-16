from pathlib import Path

import pytest
from dam.core.world import World
from dam.functions import ecs_functions
from dam.functions.mime_type_functions import set_content_mime_type
from dam.system_events.progress import ProgressCompleted
from dam_fs.commands import RegisterLocalFileCommand
from sqlalchemy import select

from dam_archive.commands import IngestArchiveMembersCommand
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
    entity_id_reg = await world.dispatch_command(register_cmd_reg).get_one_value()

    async with world.db_session_maker() as session:
        await set_content_mime_type(session, entity_id_reg, "application/zip")
        await session.commit()

    # 2. Run the extraction command
    ingest_cmd_reg = IngestArchiveMembersCommand(entity_id=entity_id_reg, depth=0)
    async with world.transaction():
        stream = world.dispatch_command(ingest_cmd_reg)
        events = [event async for event in stream]
        assert any(isinstance(event, ProgressCompleted) for event in events)

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
    entity_id_prot = await world.dispatch_command(register_cmd_prot).get_one_value()

    async with world.db_session_maker() as session:
        await set_content_mime_type(session, entity_id_prot, "application/zip")
        await session.commit()

    # 2. Run the extraction command with the correct password
    ingest_cmd_prot = IngestArchiveMembersCommand(entity_id=entity_id_prot, depth=0, passwords=["password"])
    async with world.transaction():
        stream = world.dispatch_command(ingest_cmd_prot)
        events = [event async for event in stream]
        assert any(isinstance(event, ProgressCompleted) for event in events)

    # 3. Verify the results for the protected archive
    async with world.db_session_maker() as session:
        info_prot = await ecs_functions.get_component(session, entity_id_prot, ArchiveInfoComponent)
        assert info_prot is not None
        assert info_prot.file_count == 2

        members_prot = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id_prot)
        )
        assert len(members_prot.scalars().all()) == 2


@pytest.mark.serial
@pytest.mark.asyncio
async def test_skip_already_extracted(test_world_alpha: World, test_archives: tuple[Path, Path]) -> None:
    """
    Tests that the IngestArchiveMembersCommand skips archives that have already been processed.
    """
    world = test_world_alpha
    regular_archive_path, _ = test_archives

    # 1. Register the archive file
    register_cmd = RegisterLocalFileCommand(file_path=regular_archive_path)
    entity_id = await world.dispatch_command(register_cmd).get_one_value()

    async with world.db_session_maker() as session:
        await set_content_mime_type(session, entity_id, "application/zip")
        await session.commit()

    # 2. Run the extraction command for the first time
    ingest_cmd1 = IngestArchiveMembersCommand(entity_id=entity_id, depth=0)
    async with world.transaction():
        stream1 = world.dispatch_command(ingest_cmd1)
        events1 = [event async for event in stream1]
        assert any(isinstance(event, ProgressCompleted) for event in events1)

    # 3. Verify that it was processed
    async with world.db_session_maker() as session:
        info1 = await ecs_functions.get_component(session, entity_id, ArchiveInfoComponent)
        assert info1 is not None
        assert info1.file_count == 2

    # 4. Run the extraction command for the second time
    ingest_cmd2 = IngestArchiveMembersCommand(entity_id=entity_id, depth=0)
    async with world.transaction():
        stream2 = world.dispatch_command(ingest_cmd2)
        events2 = [event async for event in stream2]
        completed_event = next((e for e in events2 if isinstance(e, ProgressCompleted)), None)
        assert completed_event is not None
        assert completed_event.message == "Already processed."

    # 5. Verify that it was skipped (no new components were created)
    # We can't easily check the logs here, so we will check that the number of members is still the same.
    async with world.db_session_maker() as session:
        members = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id)
        )
        assert len(members.scalars().all()) == 2
