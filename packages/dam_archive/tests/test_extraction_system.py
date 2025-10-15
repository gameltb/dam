"""Tests for the archive extraction system."""

import io
import zipfile
from datetime import datetime
from pathlib import Path

import pytest
from dam.commands.asset_commands import GetAssetStreamCommand, GetOrCreateEntityFromStreamCommand
from dam.core.transaction import WorldTransaction
from dam.core.types import CallableStreamProvider
from dam.core.world import World
from dam.functions import ecs_functions
from dam.functions.mime_type_functions import set_content_mime_type
from dam.system_events.entity_events import NewEntityCreatedEvent
from dam.system_events.progress import ProgressCompleted
from dam.utils.stream_utils import ChainedStream
from dam_fs.commands import RegisterLocalFileCommand
from pytest_mock import MockerFixture
from sqlalchemy import select

from dam_archive.commands.ingestion import (
    IngestArchiveCommand,
    ReissueArchiveMemberEventsCommand,
)
from dam_archive.models import ArchiveInfoComponent, ArchiveMemberComponent


@pytest.mark.serial
@pytest.mark.asyncio
async def test_ingestion_with_memory_limit_and_filename(
    test_world_alpha: World, tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test the ingestion logic with memory constraints and verifies the filename event field."""
    world = test_world_alpha

    # 1. Create a test archive with a single file
    file_content = b"a" * (1024 * 1024)  # 1 MB
    file_name_in_archive = "large_file.txt"
    archive_path = tmp_path / "large_archive.zip"

    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(file_name_in_archive, file_content)

    # 2. Register the archive
    register_cmd = RegisterLocalFileCommand(file_path=archive_path)
    entity_id = await world.dispatch_command(register_cmd).get_one_value()
    assert entity_id is not None
    tm = world.get_context(WorldTransaction)
    async with tm() as transaction:
        session = transaction.session
        await set_content_mime_type(session, entity_id, "application/zip")
        await session.commit()

    # 3. Test Case: Memory limit is not reached
    mock_memory = mocker.MagicMock()
    mock_memory.available = 2 * 1024 * 1024  # 2 MB, more than the file size

    with mocker.patch("dam_archive.systems.ingestion.psutil.virtual_memory", return_value=mock_memory):
        ingest_cmd = IngestArchiveCommand(entity_id=entity_id)
        stream = world.dispatch_command(ingest_cmd)
        events = [event async for event in stream]

        # Verify NewEntityCreatedEvent
        new_entity_event = next((e for e in events if isinstance(e, NewEntityCreatedEvent)), None)
        assert new_entity_event is not None
        assert new_entity_event.stream_provider is not None
        async with new_entity_event.stream_provider.get_stream() as stream:
            assert stream is not None
        assert new_entity_event.filename == file_name_in_archive


async def _run_ingestion_with_memory_limit(
    world: World, entity_id: int, mock_memory: MockerFixture, mocker: MockerFixture
):
    mocker.patch("dam_archive.systems.ingestion.psutil.virtual_memory", return_value=mock_memory)
    dispatch_spy = mocker.patch.object(world, "dispatch_command", wraps=world.dispatch_command)

    ingest_cmd_limit = IngestArchiveCommand(entity_id=entity_id)
    stream = world.dispatch_command(ingest_cmd_limit)
    events = [event async for event in stream]

    new_entity_event = next((e for e in events if isinstance(e, NewEntityCreatedEvent)), None)
    assert new_entity_event is not None
    assert new_entity_event.stream_provider is None

    get_or_create_cmd_call = next(
        (call for call in dispatch_spy.call_args_list if isinstance(call.args[0], GetOrCreateEntityFromStreamCommand)),
        None,
    )
    assert get_or_create_cmd_call is not None
    get_or_create_cmd = get_or_create_cmd_call.args[0]
    assert isinstance(get_or_create_cmd.stream_provider, CallableStreamProvider)
    async with get_or_create_cmd.stream_provider.get_stream() as stream:
        assert isinstance(stream, ChainedStream)


async def _run_ingestion_without_memory_limit(
    world: World, entity_id: int, file_content: bytes, mock_memory: MockerFixture, mocker: MockerFixture
):
    mocker.patch("dam_archive.systems.ingestion.psutil.virtual_memory", return_value=mock_memory)
    dispatch_spy = mocker.patch.object(world, "dispatch_command", wraps=world.dispatch_command)

    ingest_cmd_no_limit = IngestArchiveCommand(entity_id=entity_id)
    stream = world.dispatch_command(ingest_cmd_no_limit)
    events = [event async for event in stream]

    new_entity_event = next((e for e in events if isinstance(e, NewEntityCreatedEvent)), None)
    assert new_entity_event is not None
    assert new_entity_event.stream_provider is not None
    async with new_entity_event.stream_provider.get_stream() as stream:
        assert stream is not None
        assert stream.read() == file_content
        stream.seek(0)

    get_or_create_cmd_call = next(
        (call for call in dispatch_spy.call_args_list if isinstance(call.args[0], GetOrCreateEntityFromStreamCommand)),
        None,
    )
    assert get_or_create_cmd_call is not None
    get_or_create_cmd = get_or_create_cmd_call.args[0]
    assert isinstance(get_or_create_cmd.stream_provider, CallableStreamProvider)
    async with get_or_create_cmd.stream_provider.get_stream() as stream:
        assert isinstance(stream, io.BytesIO)
        stream.seek(0)
        assert stream.read() == file_content


@pytest.mark.serial
@pytest.mark.asyncio
async def test_ingestion_with_memory_limit(test_world_alpha: World, tmp_path: Path, mocker: MockerFixture) -> None:
    """Test the ingestion logic with memory constraints."""
    world = test_world_alpha
    file_content = b"a" * (1024 * 1024)  # 1 MB
    archive_path = tmp_path / "large_archive.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("large_file.txt", file_content)

    register_cmd = RegisterLocalFileCommand(file_path=archive_path)
    entity_id = await world.dispatch_command(register_cmd).get_one_value()
    assert entity_id is not None
    tm = world.get_context(WorldTransaction)
    async with tm() as transaction:
        session = transaction.session
        await set_content_mime_type(session, entity_id, "application/zip")
        await session.commit()

    mock_memory_limited = mocker.MagicMock()
    mock_memory_limited.available = 512 * 1024
    await _run_ingestion_with_memory_limit(world, entity_id, mock_memory_limited, mocker)

    async with tm() as transaction:
        session = transaction.session
        info = await ecs_functions.get_component(session, entity_id, ArchiveInfoComponent)
        if info:
            await session.delete(info)
        members = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id)
        )
        for member in members.scalars().all():
            await session.delete(member)
        await session.commit()

    mock_memory_unlimited = mocker.MagicMock()
    mock_memory_unlimited.available = 2 * 1024 * 1024
    await _run_ingestion_without_memory_limit(world, entity_id, file_content, mock_memory_unlimited, mocker)


@pytest.mark.serial
@pytest.mark.asyncio
async def test_extract_archives(test_world_alpha: World, test_archives: tuple[Path, Path]) -> None:
    """Test extracting both regular and protected archives using the new fixture."""
    world = test_world_alpha
    regular_archive_path, protected_archive_path = test_archives
    num_files_in_archive = 2

    # --- Test Regular Archive ---
    # 1. Register the regular archive file
    register_cmd_reg = RegisterLocalFileCommand(file_path=regular_archive_path)
    entity_id_reg = await world.dispatch_command(register_cmd_reg).get_one_value()
    assert entity_id_reg is not None

    tm = world.get_context(WorldTransaction)
    async with tm() as transaction:
        session = transaction.session
        await set_content_mime_type(session, entity_id_reg, "application/zip")
        await session.commit()

    # 2. Run the extraction command
    ingest_cmd_reg = IngestArchiveCommand(entity_id=entity_id_reg)
    stream = world.dispatch_command(ingest_cmd_reg)
    events = [event async for event in stream]
    assert any(isinstance(event, ProgressCompleted) for event in events)

    # 3. Verify the results for the regular archive
    async with tm() as transaction:
        session = transaction.session
        info_reg = await ecs_functions.get_component(session, entity_id_reg, ArchiveInfoComponent)
        assert info_reg is not None
        assert info_reg.comment == "regular archive comment"

        members_reg = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id_reg)
        )
        members_reg_all = members_reg.scalars().all()
        assert len(members_reg_all) == num_files_in_archive
        for member in members_reg_all:
            assert isinstance(member.modified_at, datetime)
            assert member.compressed_size is not None
            assert member.compressed_size > 0

    # --- Test Protected Archive ---
    # 1. Register the protected archive file
    register_cmd_prot = RegisterLocalFileCommand(file_path=protected_archive_path)
    entity_id_prot = await world.dispatch_command(register_cmd_prot).get_one_value()
    assert entity_id_prot is not None

    async with tm() as transaction:
        session = transaction.session
        await set_content_mime_type(session, entity_id_prot, "application/zip")
        await session.commit()

    # 2. Run the extraction command with the correct password
    ingest_cmd_prot = IngestArchiveCommand(entity_id=entity_id_prot, password="password")
    stream = world.dispatch_command(ingest_cmd_prot)
    events = [event async for event in stream]
    assert any(isinstance(event, ProgressCompleted) for event in events)

    # 3. Verify the results for the protected archive
    async with tm() as transaction:
        session = transaction.session
        info_prot = await ecs_functions.get_component(session, entity_id_prot, ArchiveInfoComponent)
        assert info_prot is not None
        assert info_prot.comment == "protected archive comment"

        members_prot = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id_prot)
        )
        members_prot_all = members_prot.scalars().all()
        assert len(members_prot_all) == num_files_in_archive
        for member in members_prot_all:
            assert isinstance(member.modified_at, datetime)
            assert member.compressed_size is not None
            assert member.compressed_size > 0


@pytest.mark.serial
@pytest.mark.asyncio
async def test_reingest_already_extracted_archive(test_world_alpha: World, test_archives: tuple[Path, Path]) -> None:
    """
    Test that IngestArchiveCommand re-issues events for already processed archives.

    This test uses the refactored _process_archive function.
    """
    world = test_world_alpha
    regular_archive_path, _ = test_archives
    num_files_in_archive = 2

    # 1. Register the archive file
    register_cmd = RegisterLocalFileCommand(file_path=regular_archive_path)
    entity_id = await world.dispatch_command(register_cmd).get_one_value()
    assert entity_id is not None

    tm = world.get_context(WorldTransaction)
    async with tm() as transaction:
        session = transaction.session
        await set_content_mime_type(session, entity_id, "application/zip")
        await session.commit()

    # 2. Run the extraction command for the first time to populate the data
    ingest_cmd1 = IngestArchiveCommand(entity_id=entity_id)
    stream1 = world.dispatch_command(ingest_cmd1)
    events1 = [event async for event in stream1]
    assert any(isinstance(event, ProgressCompleted) for event in events1)
    new_entity_events1 = [e for e in events1 if isinstance(e, NewEntityCreatedEvent)]
    assert len(new_entity_events1) == num_files_in_archive

    # 3. Verify that it was processed correctly the first time
    async with tm() as transaction:
        session = transaction.session
        info1 = await ecs_functions.get_component(session, entity_id, ArchiveInfoComponent)
        assert info1 is not None
        assert info1.comment == "regular archive comment"

        members = await session.execute(
            select(ArchiveMemberComponent).where(ArchiveMemberComponent.archive_entity_id == entity_id)
        )
        assert len(members.scalars().all()) == num_files_in_archive

    # 4. Run the re-issue command for the second time (re-ingestion)
    reissue_cmd = ReissueArchiveMemberEventsCommand(entity_id=entity_id)
    stream2 = world.dispatch_command(reissue_cmd)
    events2 = [event async for event in stream2]

    # 5. Verify that the correct events were issued for re-ingestion
    completed_event = next((e for e in events2 if isinstance(e, ProgressCompleted)), None)
    assert completed_event is not None
    assert completed_event.message == "Finished re-issuing events for members."

    new_entity_events2 = [e for e in events2 if isinstance(e, NewEntityCreatedEvent)]
    assert len(new_entity_events2) == num_files_in_archive

    # Verify the contents of the re-issued events
    filenames_in_archive = {"file1.txt", "file2.txt"}
    event_filenames = {e.filename for e in new_entity_events2}
    assert event_filenames == filenames_in_archive

    # Check that the entity IDs are the same as the first ingestion
    original_entity_ids = {e.entity_id for e in new_entity_events1}
    reingested_entity_ids = {e.entity_id for e in new_entity_events2}
    assert original_entity_ids == reingested_entity_ids

    # Check that the stream provider is None but can be retrieved on demand
    for event in new_entity_events2:
        assert event.stream_provider is None  # The event itself doesn't carry the stream

        # But we can get it via a command
        stream_cmd = GetAssetStreamCommand(entity_id=event.entity_id)
        # Use get_all_results and pick the first valid provider.
        # This is because multiple handlers can provide a stream (from the archive, from CAS),
        # and we need to consume all results to avoid deadlocks in the test runner.
        all_providers = await world.dispatch_command(stream_cmd).get_all_results()
        valid_providers = [p for p in all_providers if p is not None]
        assert valid_providers
        stream_provider = valid_providers[0]

        async with stream_provider.get_stream() as stream:
            assert stream is not None
            content = stream.read()
            assert content in (b"file one", b"file two")
