import zipfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from dam.core.world import World

from dam_app.commands import IngestAssetsCommand
from dam_app.models import ArchiveMemberComponent
from dam_app.systems.ingestion_systems import asset_ingestion_system


@pytest.fixture
def dummy_zip_file(tmp_path: Path) -> Path:
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file1.txt", b"content1")
        zf.writestr("dir/file2.txt", b"content2")
    return zip_path


@pytest.mark.asyncio
async def test_asset_ingestion_system_with_archive(dummy_zip_file: Path, mocker):
    """
    Tests the asset_ingestion_system with a zip file.
    """
    # 1. Setup
    mock_world = AsyncMock(spec=World)
    mock_world.world_config = {}  # Add the missing attribute
    mock_transaction = AsyncMock()

    # Mock create_entity_with_file to return a mock entity with an ID
    mock_archive_entity = AsyncMock()
    mock_archive_entity.id = 1

    mocker.patch(
        "dam_app.systems.ingestion_systems.dam_fs_file_operations.create_entity_with_file",
        return_value=mock_archive_entity,
    )
    mock_create_entity = mocker.patch("dam_app.systems.ingestion_systems.ecs_functions.create_entity")

    # Mock create_entity for the members of the archive
    mock_member_entity_1 = AsyncMock()
    mock_member_entity_1.id = 2
    mock_member_entity_2 = AsyncMock()
    mock_member_entity_2.id = 3
    mock_create_entity.side_effect = [mock_member_entity_1, mock_member_entity_2]

    mock_transaction.add_component_to_entity = AsyncMock()

    cmd = IngestAssetsCommand(file_paths=[str(dummy_zip_file)])

    # 2. Execute
    entity_ids = await asset_ingestion_system(cmd, mock_world, mock_transaction)

    # 3. Assert
    assert entity_ids == [1, 2, 3]

    # Check that create_entity_with_file was called for the archive
    from dam_app.systems.ingestion_systems import dam_fs_file_operations

    dam_fs_file_operations.create_entity_with_file.assert_called_once()

    # Check that create_entity was called for the two members
    assert mock_create_entity.call_count == 2

    # Check that ArchiveMemberComponent was added
    assert mock_transaction.add_component_to_entity.call_count == 2
    add_component_calls = mock_transaction.add_component_to_entity.call_args_list

    # Call 1
    assert add_component_calls[0].args[0] == 2  # entity_id
    assert isinstance(add_component_calls[0].args[1], ArchiveMemberComponent)
    assert add_component_calls[0].args[1].archive_entity_id == 1
    assert add_component_calls[0].args[1].path_in_archive == "file1.txt"

    # Call 2
    assert add_component_calls[1].args[0] == 3  # entity_id
    assert isinstance(add_component_calls[1].args[1], ArchiveMemberComponent)
    assert add_component_calls[1].args[1].archive_entity_id == 1
    assert add_component_calls[1].args[1].path_in_archive == "dir/file2.txt"
