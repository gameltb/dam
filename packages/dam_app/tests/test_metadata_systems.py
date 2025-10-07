import asyncio
import io
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dam.core.transaction import WorldTransaction
from dam.core.types import CallableStreamProvider
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
    """Fixture for a mocked WorldTransaction."""
    transaction = MagicMock(spec=WorldTransaction)
    transaction.get_components = AsyncMock(return_value=[])
    transaction.add_or_update_component = AsyncMock()
    return transaction


@pytest.fixture
def mock_world_config() -> MagicMock:
    """Fixture for a mocked WorldConfig."""
    return MagicMock()


@pytest.mark.asyncio
async def test_extract_metadata_from_file(mock_transaction: MagicMock):
    """
    Tests that metadata extraction from a file uses a persistent exiftool process
    and correctly extracts metadata.
    """
    mock_world = MagicMock()
    mock_dispatch_result = AsyncMock()
    mock_dispatch_result.get_first_non_none_value.return_value = (
        None  # No stream provider, forcing file-based extraction
    )
    mock_world.dispatch_command.return_value = mock_dispatch_result
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

            await extract_metadata_command_handler(command, mock_transaction, mock_world)

            mock_get_metadata.assert_called_once_with(filepath=temp_file)
            assert mock_transaction.add_or_update_component.call_args is not None
            added_component = mock_transaction.add_or_update_component.call_args[0][1]
            assert isinstance(added_component, ExiftoolMetadataComponent)
            assert added_component.raw_exif_json is not None
            assert added_component.raw_exif_json["FileType"] == "TXT"


@pytest.mark.asyncio
async def test_extract_metadata_from_stream(mock_transaction: MagicMock):
    """
    Tests that metadata extraction from a stream uses a persistent exiftool process
    and correctly extracts metadata without creating a temporary file.
    """
    stream_content = b"This is a test stream."
    read_content = None

    async def side_effect(stream: io.BytesIO, **kwargs: Any):
        nonlocal read_content
        read_content = stream.read()
        return {"FileType": "STREAM"}

    def stream_provider_func():
        return io.BytesIO(stream_content)

    entity_id = 1
    command = ExtractExifMetadataCommand(
        entity_id=entity_id, stream_provider=CallableStreamProvider(stream_provider_func)
    )

    # Mock the world object needed by open_stream
    mock_world = MagicMock()

    with patch.object(exiftool_instance, "get_metadata", new_callable=AsyncMock) as mock_get_metadata:
        mock_get_metadata.side_effect = side_effect

        await extract_metadata_command_handler(command, mock_transaction, mock_world)

        # Verify that get_metadata was called with a stream
        assert mock_get_metadata.call_count == 1
        assert read_content == stream_content

        assert mock_transaction.add_or_update_component.call_args is not None
        added_component = mock_transaction.add_or_update_component.call_args[0][1]
        assert isinstance(added_component, ExiftoolMetadataComponent)
        assert added_component.raw_exif_json is not None
        assert added_component.raw_exif_json["FileType"] == "STREAM"


@pytest.mark.asyncio
async def test_exiftool_process_reuse():
    """Tests that the same exiftool process is reused for multiple calls."""
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
    """Tests that the file extension is correctly extracted and passed to exiftool."""
    exiftool = ExifTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Use a jpg extension, as it's universally supported by exiftool
        temp_file = Path(temp_dir) / "test.jpg"
        # A minimal valid JPEG file content
        temp_file.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x07\x07\x08\t\x07\x06\x06\x08\x0b\t\n\n\n\n\n\n\x0c\x0b\x0c\x0b\x0b\x0c\x0b\x0b\x0b\x0b\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xeb\xbf\xff\xd9"
        )

        metadata = await exiftool.get_metadata(filepath=temp_file)

        assert metadata is not None

    await exiftool.stop()


@pytest.mark.asyncio
async def test_extract_metadata_removes_sourcefile():
    """Tests that the SourceFile field is removed from the exiftool output."""
    exiftool = ExifTool()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.txt"
        temp_file.write_text("This is a test file.")

        metadata = await exiftool.get_metadata(filepath=temp_file)

        assert metadata is not None
        assert "SourceFile" not in metadata

    await exiftool.stop()


@pytest.mark.asyncio
async def test_extract_metadata_with_large_json_output():
    """
    Tests that metadata extraction can handle large JSON output from exiftool
    without raising a LimitOverrunError.
    """
    exiftool = ExifTool()
    await exiftool.start()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        image_path = temp_dir_path / "test_image.jpg"
        # A minimal valid JPEG file content
        image_path.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x07\x07\x08\t\x07\x06\x06\x08\x0b\t\n\n\n\n\n\n\x0c\x0b\x0c\x0b\x0b\x0c\x0b\x0b\x0b\x0b\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xeb\xbf\xff\xd9"
        )

        large_comment = "A" * 100000
        comment_file_path = temp_dir_path / "large_comment.txt"
        comment_file_path.write_text(large_comment)

        large_meta_image_path = temp_dir_path / "test_image_large_meta.jpg"
        # We use -m to ignore minor errors about the size of the comment
        process = await asyncio.create_subprocess_exec(
            "exiftool",
            "-m",
            f"-UserComment<={comment_file_path}",
            str(image_path),
            "-o",
            str(large_meta_image_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            assert process.returncode is not None
            raise subprocess.CalledProcessError(
                process.returncode,
                [
                    "exiftool",
                    "-m",
                    f"-UserComment<={comment_file_path}",
                    str(image_path),
                    "-o",
                    str(large_meta_image_path),
                ],
                stdout,
                stderr,
            )

        metadata = await exiftool.get_metadata(filepath=large_meta_image_path)

        assert metadata is not None
        assert "EXIF:UserComment" in metadata
        assert len(metadata["EXIF:UserComment"]) == 100000
        assert metadata["EXIF:UserComment"] == large_comment

    await exiftool.stop()
