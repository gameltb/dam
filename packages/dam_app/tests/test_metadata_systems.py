import io
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dam.core.ecs import Entity  # type: ignore
from dam.core.transaction import EcsTransaction
from dam.models.metadata.exiftool_metadata_component import ExiftoolMetadataComponent
from dam_fs.models.file_location_component import FileLocationComponent

from dam_app.commands import ExtractExifMetadataCommand
from dam_app.systems.metadata_systems import (
    exiftool_instance,
    extract_metadata_command_handler,
)


@pytest.fixture
def mock_transaction() -> MagicMock:
    """Fixture for a mocked EcsTransaction."""
    transaction = MagicMock(spec=EcsTransaction)
    transaction.get_components = AsyncMock(return_value=[])
    transaction.add_or_update_component = AsyncMock()
    return transaction


@pytest.fixture
def mock_world_config() -> MagicMock:
    """Fixture for a mocked WorldConfig."""
    return MagicMock()


@pytest.mark.asyncio
async def test_extract_metadata_from_file(mock_transaction: MagicMock, mock_world_config: MagicMock):
    """
    Tests that metadata extraction from a file uses a persistent exiftool process
    and correctly extracts metadata.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.txt"
        temp_file.write_text("This is a test file.")

        file_location = FileLocationComponent(url=temp_file.as_uri(), last_modified_at=datetime.now())
        mock_transaction.get_components.return_value = [file_location]

        entity_id = Entity(1)
        command = ExtractExifMetadataCommand(entity_id=entity_id)

        with patch.object(exiftool_instance, "get_metadata", new_callable=AsyncMock) as mock_get_metadata:
            mock_get_metadata.return_value = {"FileType": "TXT"}

            await extract_metadata_command_handler(command, mock_transaction, mock_world_config)

            mock_get_metadata.assert_called_once_with(filepath=temp_file)
            assert mock_transaction.add_or_update_component.call_args is not None
            added_component = mock_transaction.add_or_update_component.call_args[0][1]
            assert isinstance(added_component, ExiftoolMetadataComponent)
            assert added_component.raw_exif_json["FileType"] == "TXT"


@pytest.mark.asyncio
async def test_extract_metadata_from_stream(mock_transaction: MagicMock, mock_world_config: MagicMock):
    """
    Tests that metadata extraction from a stream uses a persistent exiftool process
    and correctly extracts metadata without creating a temporary file.
    """
    stream_content = b"This is a test stream."
    stream = io.BytesIO(stream_content)

    entity_id = Entity(1)
    command = ExtractExifMetadataCommand(entity_id=entity_id, stream=stream)

    with patch.object(exiftool_instance, "get_metadata", new_callable=AsyncMock) as mock_get_metadata:
        mock_get_metadata.return_value = {"FileType": "STREAM"}

        await extract_metadata_command_handler(command, mock_transaction, mock_world_config)

        mock_get_metadata.assert_called_once_with(stream=stream)
        assert mock_transaction.add_or_update_component.call_args is not None
        added_component = mock_transaction.add_or_update_component.call_args[0][1]
        assert isinstance(added_component, ExiftoolMetadataComponent)
        assert added_component.raw_exif_json["FileType"] == "STREAM"


@pytest.mark.asyncio
async def test_exiftool_process_reuse(mock_transaction: MagicMock, mock_world_config: MagicMock):
    """
    Tests that the same exiftool process is reused for multiple calls.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.txt"
        temp_file.write_text("This is a test file.")

        file_location = FileLocationComponent(url=temp_file.as_uri(), last_modified_at=datetime.now())
        mock_transaction.get_components.return_value = [file_location]

        entity_id = Entity(1)
        command = ExtractExifMetadataCommand(entity_id=entity_id)

        with (
            patch.object(exiftool_instance, "start", new_callable=AsyncMock) as mock_start,
            patch.object(exiftool_instance, "stop", new_callable=AsyncMock) as mock_stop,
            patch.object(exiftool_instance, "get_metadata", new_callable=AsyncMock) as mock_get_metadata,
        ):
            mock_get_metadata.return_value = {"FileType": "TXT"}

            await extract_metadata_command_handler(command, mock_transaction, mock_world_config)
            await extract_metadata_command_handler(command, mock_transaction, mock_world_config)

            mock_start.assert_called()
            mock_stop.assert_not_called()
            assert mock_get_metadata.call_count == 2

        await exiftool_instance.stop()
        mock_stop.assert_called_once()
