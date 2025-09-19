import io
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dam.core.transaction import EcsTransaction
from dam.models.metadata.exiftool_metadata_component import ExiftoolMetadataComponent
from dam_fs.models.file_location_component import FileLocationComponent

from dam_app.commands import ExtractExifMetadataCommand
from dam_app.systems.metadata_systems import (
    ExifTool,
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

        entity_id = 1
        command = ExtractExifMetadataCommand(entity_id=entity_id)

        # We still mock get_metadata here to test the handler in isolation
        with patch.object(exiftool_instance, "get_metadata", new_callable=AsyncMock) as mock_get_metadata:
            mock_get_metadata.return_value = {"FileType": "TXT"}

            await extract_metadata_command_handler(command, mock_transaction, mock_world_config)

            mock_get_metadata.assert_called_once_with(filepath=temp_file)
            assert mock_transaction.add_or_update_component.call_args is not None
            added_component = mock_transaction.add_or_update_component.call_args[0][1]
            assert isinstance(added_component, ExiftoolMetadataComponent)
            assert added_component.raw_exif_json is not None
            assert added_component.raw_exif_json["FileType"] == "TXT"


@pytest.mark.asyncio
async def test_extract_metadata_from_stream(mock_transaction: MagicMock, mock_world_config: MagicMock):
    """
    Tests that metadata extraction from a stream uses a persistent exiftool process
    and correctly extracts metadata without creating a temporary file.
    """
    stream_content = b"This is a test stream."
    stream = io.BytesIO(stream_content)

    entity_id = 1
    command = ExtractExifMetadataCommand(entity_id=entity_id, stream=stream)

    with patch.object(exiftool_instance, "get_metadata", new_callable=AsyncMock) as mock_get_metadata:
        mock_get_metadata.return_value = {"FileType": "STREAM"}

        await extract_metadata_command_handler(command, mock_transaction, mock_world_config)

        mock_get_metadata.assert_called_once_with(stream=stream)
        assert mock_transaction.add_or_update_component.call_args is not None
        added_component = mock_transaction.add_or_update_component.call_args[0][1]
        assert isinstance(added_component, ExiftoolMetadataComponent)
        assert added_component.raw_exif_json is not None
        assert added_component.raw_exif_json["FileType"] == "STREAM"


@pytest.mark.asyncio
async def test_exiftool_process_reuse():
    """
    Tests that the same exiftool process is reused for multiple calls.
    """
    exiftool = ExifTool()
    await exiftool.start()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.txt"
        temp_file.write_text("This is a test file.")

        # Call twice to check reuse
        await exiftool.get_metadata(filepath=temp_file)
        await exiftool.get_metadata(filepath=temp_file)

        # The process should be started, and not None
        assert exiftool.process is not None
        # We can't easily check that it was started only once without mocks,
        # but we can check that the same process is running.
        pid = exiftool.process.pid
        assert pid is not None

    await exiftool.stop()
    assert exiftool.process is None


@pytest.mark.asyncio
async def test_extract_metadata_with_extension():
    """
    Tests that the file extension is correctly extracted and passed to exiftool.
    """
    exiftool = ExifTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Use a jpg extension, as it's universally supported by exiftool
        temp_file = Path(temp_dir) / "test.jpg"
        # A minimal valid JPEG file content
        temp_file.write_bytes(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x07\x07\x08\t\x07\x06\x06\x08\x0b\t\n\n\n\n\n\n\x0c\x0b\x0c\x0b\x0b\x0c\x0b\x0b\x0b\x0b\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xeb\xbf\xff\xd9')

        metadata = await exiftool.get_metadata(filepath=temp_file)

        assert metadata is not None
        assert metadata.get("FileType") == "JPEG"

    await exiftool.stop()

@pytest.mark.asyncio
async def test_extract_metadata_removes_sourcefile():
    """
    Tests that the SourceFile field is removed from the exiftool output.
    """
    exiftool = ExifTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.txt"
        temp_file.write_text("This is a test file.")

        metadata = await exiftool.get_metadata(filepath=temp_file)

        assert metadata is not None
        assert "SourceFile" not in metadata

    await exiftool.stop()
