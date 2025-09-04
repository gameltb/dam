from unittest.mock import AsyncMock

import pytest
from dam.commands import GetAssetFilenamesCommand

from dam_archive.models import ArchiveMemberComponent
from dam_archive.systems import get_archive_asset_filenames_handler


@pytest.mark.asyncio
async def test_get_archive_asset_filenames_handler_with_filename():
    """
    Tests that the handler returns the filename when an ArchiveMemberComponent exists.
    """
    entity_id = 1
    path_in_archive = "path/to/file.jpg"

    mock_transaction = AsyncMock()
    mock_transaction.get_component.return_value = ArchiveMemberComponent(
        archive_entity_id=99, path_in_archive=path_in_archive
    )

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_archive_asset_filenames_handler(command, mock_transaction)

    assert result == [path_in_archive]
    mock_transaction.get_component.assert_called_once_with(entity_id, ArchiveMemberComponent)


@pytest.mark.asyncio
async def test_get_archive_asset_filenames_handler_no_component():
    """
    Tests that the handler returns None when no ArchiveMemberComponent exists.
    """
    entity_id = 1

    mock_transaction = AsyncMock()
    mock_transaction.get_component.return_value = None

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_archive_asset_filenames_handler(command, mock_transaction)

    assert result is None
    mock_transaction.get_component.assert_called_once_with(entity_id, ArchiveMemberComponent)


@pytest.mark.asyncio
async def test_get_archive_asset_filenames_handler_no_filename():
    """
    Tests that the handler returns None when the component exists but has no path.
    """
    entity_id = 1

    mock_transaction = AsyncMock()
    mock_transaction.get_component.return_value = ArchiveMemberComponent(archive_entity_id=99, path_in_archive=None)

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_archive_asset_filenames_handler(command, mock_transaction)

    assert result is None
    mock_transaction.get_component.assert_called_once_with(entity_id, ArchiveMemberComponent)
