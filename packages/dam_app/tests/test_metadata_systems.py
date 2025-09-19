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
@patch("dam_app.systems.metadata_systems.shutil.which", return_value="/usr/bin/exiftool")
@patch("asyncio.create_subprocess_exec")
async def test_exiftool_process_reuse(mock_exec: AsyncMock, mock_which: MagicMock):
    """
    Tests that the same exiftool process is reused for multiple calls.
    """
    # Setup mock process
    mock_process = AsyncMock()
    mock_process.returncode = None
    mock_process.stdin.drain = AsyncMock()
    mock_process.stdout.readuntil = AsyncMock(return_value=b'[{ "FileType": "TXT" }]{ready}\n')
    mock_exec.return_value = mock_process

    exiftool = ExifTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.txt"
        temp_file.write_text("This is a test file.")

        await exiftool.get_metadata(filepath=temp_file)
        await exiftool.get_metadata(filepath=temp_file)

        mock_exec.assert_called_once()
        assert mock_process.stdin.write.call_count == 2

    await exiftool.stop()


@pytest.mark.asyncio
@patch("dam_app.systems.metadata_systems.shutil.which", return_value="/usr/bin/exiftool")
@patch("asyncio.create_subprocess_exec")
async def test_extract_metadata_with_extension(mock_exec: AsyncMock, mock_which: MagicMock):
    """
    Tests that the file extension is correctly extracted and passed to exiftool.
    """
    mock_process = AsyncMock()
    mock_process.returncode = None
    mock_process.stdin.drain = AsyncMock()
    mock_process.stdout.readuntil = AsyncMock(return_value=b'[{ "FileType": "JPG" }]{ready}\n')
    mock_exec.return_value = mock_process

    exiftool = ExifTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.jpg"
        temp_file.write_text("This is a test file.")

        await exiftool.get_metadata(filepath=temp_file)

        written_command = mock_process.stdin.write.call_args[0][0]
        assert b"-ext\njpg" in written_command

    await exiftool.stop()


@pytest.mark.asyncio
@patch("dam_app.systems.metadata_systems.shutil.which", return_value="/usr/bin/exiftool")
@patch("asyncio.create_subprocess_exec")
async def test_extract_metadata_removes_sourcefile(mock_exec: AsyncMock, mock_which: MagicMock):
    """
    Tests that the SourceFile field is removed from the exiftool output.
    """
    mock_process = AsyncMock()
    mock_process.returncode = None
    mock_process.stdin.drain = AsyncMock()
    mock_process.stdout.readuntil = AsyncMock(
        return_value=b'[{ "SourceFile": "test.jpg", "FileType": "JPG" }]{ready}\n'
    )
    mock_exec.return_value = mock_process

    exiftool = ExifTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.jpg"
        temp_file.write_text("This is a test file.")

        metadata = await exiftool.get_metadata(filepath=temp_file)

        assert metadata is not None
        assert "SourceFile" not in metadata
        assert metadata["FileType"] == "JPG"

    await exiftool.stop()
