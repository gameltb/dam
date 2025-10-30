"""Tests for the get_archive_asset_filenames_handler system."""

import pytest
from dam.commands.asset_commands import GetAssetFilenamesCommand
from pytest_mock import MockerFixture

from dam_archive.models import ArchiveMemberComponent
from dam_archive.systems.ingestion import get_archive_asset_filenames_handler


@pytest.mark.asyncio
async def test_get_archive_asset_filenames_handler_with_filename(mocker: MockerFixture) -> None:
    """Test that the handler returns the filename when an ArchiveMemberComponent exists."""
    entity_id = 1
    path_in_archive = "path/to/file.jpg"

    mock_transaction = mocker.AsyncMock()
    mock_transaction.get_components.return_value = [
        ArchiveMemberComponent(
            archive_entity_id=99,
            path_in_archive=path_in_archive,
            modified_at=None,
            compressed_size=None,
            tree_entity_id=1,
            node_id=1,
        )
    ]

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_archive_asset_filenames_handler(command, mock_transaction)

    assert result == [path_in_archive]
    mock_transaction.get_components.assert_called_once_with(entity_id, ArchiveMemberComponent)


@pytest.mark.asyncio
async def test_get_archive_asset_filenames_handler_no_component(mocker: MockerFixture) -> None:
    """Test that the handler returns None when no ArchiveMemberComponent exists."""
    entity_id = 1

    mock_transaction = mocker.AsyncMock()
    mock_transaction.get_components.return_value = []

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_archive_asset_filenames_handler(command, mock_transaction)

    assert result is None
    mock_transaction.get_components.assert_called_once_with(entity_id, ArchiveMemberComponent)


@pytest.mark.asyncio
async def test_get_archive_asset_filenames_handler_no_filename(mocker: MockerFixture) -> None:
    """Test that the handler returns an empty string when the component exists but has no path."""
    entity_id = 1

    mock_transaction = mocker.AsyncMock()
    mock_transaction.get_components.return_value = [
        ArchiveMemberComponent(
            archive_entity_id=99,
            path_in_archive="",
            modified_at=None,
            compressed_size=None,
            tree_entity_id=1,
            node_id=1,
        )
    ]

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_archive_asset_filenames_handler(command, mock_transaction)

    assert result == [""]
    mock_transaction.get_components.assert_called_once_with(entity_id, ArchiveMemberComponent)
