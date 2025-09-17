import io
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dam.core.world import World
from dam.functions import ecs_functions
from dam.functions.mime_type_functions import set_content_mime_type
from dam.system_events import NewEntityCreatedEvent
from dam.system_events.progress import ProgressCompleted
from dam.utils.stream_utils import ChainedStream
from dam_fs.commands import RegisterLocalFileCommand
from sqlalchemy import select

from dam_archive.commands import IngestArchiveCommand
from dam_archive.models import ArchiveInfoComponent, ArchiveMemberComponent


@pytest.mark.serial
@pytest.mark.asyncio
async def test_ingest_archive_with_stream(test_world_alpha: World, tmp_path: Path) -> None:
    """
    Tests that the IngestArchiveCommand can be called with a stream directly.
    """
    world = test_world_alpha

    # 1. Create an in-memory archive
    file_content = b"a" * 1024
    file_name_in_archive = "test_file.txt"
    archive_stream = io.BytesIO()
    import zipfile

    with zipfile.ZipFile(archive_stream, "w") as zf:
        zf.writestr(file_name_in_archive, file_content)
    archive_stream.seek(0)

    # 2. Register a dummy entity for the archive
    # In a real scenario, this entity would already exist.
    async with world.db_session_maker() as session:
        entity = await ecs_functions.create_entity(session)
        await set_content_mime_type(session, entity.id, "application/zip")
        await session.commit()
        entity_id = entity.id

    # 3. Run the extraction command with the stream
    ingest_cmd = IngestArchiveCommand(entity_id=entity_id, depth=0, stream=archive_stream)
    async with world.transaction():
        stream = world.dispatch_command(ingest_cmd)
        events = [event async for event in stream]
        assert any(isinstance(event, ProgressCompleted) for event in events)

    # 4. Verify the results
    async with world.db_session_maker() as session:
        info = await ecs_functions.get_component(session, entity_id, ArchiveInfoComponent)
        assert info is not None
        members = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id)
        )
        for member in members.scalars().all():
            assert isinstance(member.modified_at, datetime)


@pytest.mark.serial
@pytest.mark.asyncio
async def test_ingestion_with_memory_limit_and_filename(test_world_alpha: World, tmp_path: Path) -> None:
    """
    Tests the ingestion logic with memory constraints and verifies the filename event field.
    """
    world = test_world_alpha

    # 1. Create a test archive with a single file
    file_content = b"a" * (1024 * 1024)  # 1 MB
    file_name_in_archive = "large_file.txt"
    archive_path = tmp_path / "large_archive.zip"
    import zipfile

    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(file_name_in_archive, file_content)

    # 2. Register the archive
    register_cmd = RegisterLocalFileCommand(file_path=archive_path)
    entity_id = await world.dispatch_command(register_cmd).get_one_value()
    async with world.db_session_maker() as session:
        await set_content_mime_type(session, entity_id, "application/zip")
        await session.commit()

    # 3. Test Case: Memory limit is not reached
    mock_memory = MagicMock()
    mock_memory.available = 2 * 1024 * 1024  # 2 MB, more than the file size

    with patch("dam_archive.systems.psutil.virtual_memory", return_value=mock_memory):
        ingest_cmd = IngestArchiveCommand(entity_id=entity_id, depth=0)
        async with world.transaction():
            stream = world.dispatch_command(ingest_cmd)
            events = [event async for event in stream]

            # Verify NewEntityCreatedEvent
            new_entity_event = next((e for e in events if isinstance(e, NewEntityCreatedEvent)), None)
            assert new_entity_event is not None
            assert new_entity_event.file_stream is not None
            assert new_entity_event.filename == file_name_in_archive


@pytest.mark.serial
@pytest.mark.asyncio
async def test_ingestion_with_memory_limit(test_world_alpha: World, tmp_path: Path) -> None:
    """
    Tests the ingestion logic with memory constraints.
    """
    world = test_world_alpha

    # 1. Create a test archive with a single file
    file_content = b"a" * (1024 * 1024)  # 1 MB
    archive_path = tmp_path / "large_archive.zip"
    import zipfile

    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("large_file.txt", file_content)

    # 2. Register the archive
    register_cmd = RegisterLocalFileCommand(file_path=archive_path)
    entity_id = await world.dispatch_command(register_cmd).get_one_value()
    async with world.db_session_maker() as session:
        await set_content_mime_type(session, entity_id, "application/zip")
        await session.commit()

    # 3. Test Case 1: Memory limit is reached
    mock_memory = MagicMock()
    mock_memory.available = 512 * 1024  # 512 KB, less than the file size

    with (
        patch("dam_archive.systems.psutil.virtual_memory", return_value=mock_memory),
        patch.object(world, "dispatch_command", wraps=world.dispatch_command) as dispatch_spy,
    ):
        ingest_cmd_limit = IngestArchiveCommand(entity_id=entity_id, depth=0)
        async with world.transaction():
            stream = world.dispatch_command(ingest_cmd_limit)
            events = [event async for event in stream]

            # Verify NewEntityCreatedEvent
            new_entity_event = next((e for e in events if isinstance(e, NewEntityCreatedEvent)), None)
            assert new_entity_event is not None
            assert new_entity_event.file_stream is None

            # Verify stream passed to GetOrCreateEntityFromStreamCommand
            get_or_create_cmd = dispatch_spy.call_args.args[0]
            assert isinstance(get_or_create_cmd.stream, ChainedStream)

    # 4. Clean up components for next run
    async with world.db_session_maker() as session:
        info = await ecs_functions.get_component(session, entity_id, ArchiveInfoComponent)
        if info:
            await session.delete(info)
        members = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id)
        )
        for member in members.scalars().all():
            await session.delete(member)
        await session.commit()

    # 5. Test Case 2: Memory limit is not reached
    mock_memory.available = 2 * 1024 * 1024  # 2 MB, more than the file size

    with (
        patch("dam_archive.systems.psutil.virtual_memory", return_value=mock_memory),
        patch.object(world, "dispatch_command", wraps=world.dispatch_command) as dispatch_spy,
    ):
        ingest_cmd_no_limit = IngestArchiveCommand(entity_id=entity_id, depth=0)
        async with world.transaction():
            stream = world.dispatch_command(ingest_cmd_no_limit)
            events = [event async for event in stream]

            # Verify NewEntityCreatedEvent
            new_entity_event = next((e for e in events if isinstance(e, NewEntityCreatedEvent)), None)
            assert new_entity_event is not None
            assert new_entity_event.file_stream is not None
            assert new_entity_event.file_stream.read() == file_content
            new_entity_event.file_stream.seek(0)

            # Verify stream passed to GetOrCreateEntityFromStreamCommand
            get_or_create_cmd = dispatch_spy.call_args.args[0]
            assert isinstance(get_or_create_cmd.stream, io.BytesIO)
            assert get_or_create_cmd.stream.read() == file_content


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
    ingest_cmd_reg = IngestArchiveCommand(entity_id=entity_id_reg, depth=0)
    async with world.transaction():
        stream = world.dispatch_command(ingest_cmd_reg)
        events = [event async for event in stream]
        assert any(isinstance(event, ProgressCompleted) for event in events)

    # 3. Verify the results for the regular archive
    async with world.db_session_maker() as session:
        info_reg = await ecs_functions.get_component(session, entity_id_reg, ArchiveInfoComponent)
        assert info_reg is not None
        assert info_reg.comment == "regular archive comment"

        members_reg = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id_reg)
        )
        members_reg_all = members_reg.scalars().all()
        assert len(members_reg_all) == 2
        for member in members_reg_all:
            assert isinstance(member.modified_at, datetime)

    # --- Test Protected Archive ---
    # 1. Register the protected archive file
    register_cmd_prot = RegisterLocalFileCommand(file_path=protected_archive_path)
    entity_id_prot = await world.dispatch_command(register_cmd_prot).get_one_value()

    async with world.db_session_maker() as session:
        await set_content_mime_type(session, entity_id_prot, "application/zip")
        await session.commit()

    # 2. Run the extraction command with the correct password
    ingest_cmd_prot = IngestArchiveCommand(entity_id=entity_id_prot, depth=0, passwords=["password"])
    async with world.transaction():
        stream = world.dispatch_command(ingest_cmd_prot)
        events = [event async for event in stream]
        assert any(isinstance(event, ProgressCompleted) for event in events)

    # 3. Verify the results for the protected archive
    async with world.db_session_maker() as session:
        info_prot = await ecs_functions.get_component(session, entity_id_prot, ArchiveInfoComponent)
        assert info_prot is not None
        assert info_prot.comment == "protected archive comment"

        members_prot = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id_prot)
        )
        members_prot_all = members_prot.scalars().all()
        assert len(members_prot_all) == 2
        for member in members_prot_all:
            assert isinstance(member.modified_at, datetime)


@pytest.mark.serial
@pytest.mark.asyncio
async def test_skip_already_extracted(test_world_alpha: World, test_archives: tuple[Path, Path]) -> None:
    """
    Tests that the IngestArchiveCommand skips archives that have already been processed.
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
    ingest_cmd1 = IngestArchiveCommand(entity_id=entity_id, depth=0)
    async with world.transaction():
        stream1 = world.dispatch_command(ingest_cmd1)
        events1 = [event async for event in stream1]
        assert any(isinstance(event, ProgressCompleted) for event in events1)

    # 3. Verify that it was processed
    async with world.db_session_maker() as session:
        info1 = await ecs_functions.get_component(session, entity_id, ArchiveInfoComponent)
        assert info1 is not None
        assert info1.comment == "regular archive comment"

    # 4. Run the extraction command for the second time
    ingest_cmd2 = IngestArchiveCommand(entity_id=entity_id, depth=0)
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
