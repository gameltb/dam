from pathlib import Path

import pytest
from dam.core.world import World
from dam.functions import ecs_functions
from dam_fs.commands import RegisterLocalFileCommand

from dam_archive.commands import ExtractArchiveCommand
from dam_archive.models import ArchiveInfoComponent, ArchiveMemberComponent


@pytest.fixture
def regular_archive_path() -> Path:
    return Path(__file__).parent / "test_assets" / "regular_archive.zip"


@pytest.fixture
def protected_archive_path() -> Path:
    return Path(__file__).parent / "test_assets" / "protected_archive.zip"


@pytest.mark.serial
@pytest.mark.asyncio
async def test_extract_regular_archive(test_world_alpha: World, regular_archive_path: Path):
    """
    Tests extracting a regular, non-encrypted archive.
    """
    world = test_world_alpha

    # 1. Register the archive file
    register_cmd = RegisterLocalFileCommand(file_path=regular_archive_path)
    cmd_result = await world.dispatch_command(register_cmd)
    entity_id = cmd_result.get_one_value()

    # 2. Run the extraction command
    extract_cmd = ExtractArchiveCommand(entity_id=entity_id)
    await world.dispatch_command(extract_cmd)

    # 3. Verify the results
    async with world.db_session_maker() as session:
        # Check that the archive is marked as processed
        info = await ecs_functions.get_component(session, entity_id, ArchiveInfoComponent)
        assert info is not None
        assert info.file_count == 2

        # Find the members
        members = await ecs_functions.find_entities_with_components(session, [ArchiveMemberComponent])
        assert len(members) == 2

        # Check that the members are linked correctly
        member_paths: set[str] = set()
        for member_entity in members:
            member_comp = await ecs_functions.get_component(session, member_entity.id, ArchiveMemberComponent)
            assert member_comp is not None
            assert member_comp.archive_entity_id == entity_id
            member_paths.add(member_comp.path_in_archive)

        assert "test_archive_content/file1.txt" in member_paths
        assert "test_archive_content/file2.txt" in member_paths


@pytest.mark.serial
@pytest.mark.asyncio
async def test_extract_protected_archive(test_world_alpha: World, protected_archive_path: Path):
    """
    Tests extracting a password-protected archive.
    """
    world = test_world_alpha

    # 1. Register the archive file
    register_cmd = RegisterLocalFileCommand(file_path=protected_archive_path)
    cmd_result = await world.dispatch_command(register_cmd)
    entity_id = cmd_result.get_one_value()

    # 2. Run the extraction command with the correct password
    extract_cmd = ExtractArchiveCommand(entity_id=entity_id, passwords=["password"])
    await world.dispatch_command(extract_cmd)

    # 3. Verify the results
    async with world.db_session_maker() as session:
        # Check that the archive is marked as processed
        info = await ecs_functions.get_component(session, entity_id, ArchiveInfoComponent)
        assert info is not None
        assert info.file_count == 2
