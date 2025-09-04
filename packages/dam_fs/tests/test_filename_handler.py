from unittest.mock import AsyncMock

import pytest
from dam.commands import GetAssetFilenamesCommand

from dam_fs.models import FilePropertiesComponent
from dam_fs.systems.asset_lifecycle_systems import get_fs_asset_filenames_handler


@pytest.mark.asyncio
async def test_get_fs_asset_filenames_handler_with_filename():
    """
    Tests that the handler returns the filename when a FilePropertiesComponent exists.
    """
    entity_id = 1
    filename = "test.jpg"

    mock_transaction = AsyncMock()
    mock_transaction.get_component.return_value = FilePropertiesComponent(
        original_filename=filename, file_size_bytes=123
    )

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_fs_asset_filenames_handler(command, mock_transaction)

    assert result == [filename]
    mock_transaction.get_component.assert_called_once_with(entity_id, FilePropertiesComponent)


@pytest.mark.asyncio
async def test_get_fs_asset_filenames_handler_no_component():
    """
    Tests that the handler returns None when no FilePropertiesComponent exists.
    """
    entity_id = 1

    mock_transaction = AsyncMock()
    mock_transaction.get_component.return_value = None

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_fs_asset_filenames_handler(command, mock_transaction)

    assert result is None
    mock_transaction.get_component.assert_called_once_with(entity_id, FilePropertiesComponent)


@pytest.mark.asyncio
async def test_get_fs_asset_filenames_handler_no_filename():
    """
    Tests that the handler returns None when the component exists but has no filename.
    """
    entity_id = 1

    mock_transaction = AsyncMock()
    mock_transaction.get_component.return_value = FilePropertiesComponent(original_filename=None, file_size_bytes=123)

    command = GetAssetFilenamesCommand(entity_id=entity_id)

    result = await get_fs_asset_filenames_handler(command, mock_transaction)

    assert result is None
    mock_transaction.get_component.assert_called_once_with(entity_id, FilePropertiesComponent)
