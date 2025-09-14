from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from dam.commands import GetAssetFilenamesCommand

from dam_fs.models import FilenameComponent
from dam_fs.systems.asset_lifecycle_systems import get_fs_asset_filenames_handler


@pytest.mark.asyncio
async def test_get_fs_asset_filenames_handler_with_filename() -> None:
    """
    Tests that the handler returns the filename when a FilenameComponent exists.
    """
    entity_id = 1
    filename = "test.jpg"
    now = datetime.now(timezone.utc)

    mock_transaction = AsyncMock()
    mock_transaction.get_components.return_value = [FilenameComponent(filename=filename, first_seen_at=now)]

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_fs_asset_filenames_handler(command, mock_transaction)

    assert result == [filename]
    mock_transaction.get_components.assert_called_once_with(entity_id, FilenameComponent)


@pytest.mark.asyncio
async def test_get_fs_asset_filenames_handler_no_component() -> None:
    """
    Tests that the handler returns None when no FilenameComponent exists.
    """
    entity_id = 1

    mock_transaction = AsyncMock()
    mock_transaction.get_components.return_value = []

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_fs_asset_filenames_handler(command, mock_transaction)

    assert result is None
    mock_transaction.get_components.assert_called_once_with(entity_id, FilenameComponent)


@pytest.mark.asyncio
async def test_get_fs_asset_filenames_handler_no_filename() -> None:
    """
    Tests that the handler returns an empty list when the component exists but has no filename.
    """
    entity_id = 1
    now = datetime.now(timezone.utc)

    mock_transaction = AsyncMock()
    mock_transaction.get_components.return_value = [FilenameComponent(filename=None, first_seen_at=now)]

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_fs_asset_filenames_handler(command, mock_transaction)

    assert result == []
    mock_transaction.get_components.assert_called_once_with(entity_id, FilenameComponent)
