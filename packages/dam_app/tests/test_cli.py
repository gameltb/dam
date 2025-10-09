"""Tests for the DAM application's CLI commands."""

import io
from pathlib import Path
from typing import Any

import pytest
from dam.commands.asset_commands import GetAssetFilenamesCommand, GetMimeTypeCommand
from dam.commands.core import BaseCommand
from dam.core.config import Settings
from dam.core.operations import AssetOperation
from dam.core.types import CallableStreamProvider
from dam.core.world import World
from dam.system_events.entity_events import NewEntityCreatedEvent
from dam_archive.commands import CheckArchiveCommand, IngestArchiveCommand
from dam_fs.commands import FindEntityByFilePropertiesCommand, RegisterLocalFileCommand
from pytest_mock import MockerFixture

from dam_app.cli.assets import add_assets, check_data, process_entities, remove_data
from dam_app.commands import CheckExifMetadataCommand, ExtractExifMetadataCommand
from dam_app.main import (
    cli_list_worlds,
    create_and_register_all_worlds_from_settings,
)


@pytest.mark.serial
def test_cli_list_worlds(settings_override: Settings, capsys: pytest.CaptureFixture[Any]):
    """Test the list-worlds command."""
    # Ensure worlds are registered
    create_and_register_all_worlds_from_settings(app_settings=settings_override)

    cli_list_worlds()

    captured = capsys.readouterr()
    assert "test_world_alpha" in captured.out
    assert "test_world_beta" in captured.out


@pytest.mark.asyncio
@pytest.mark.usefixtures("capsys")
async def test_add_assets_with_recursive_process_option(tmp_path: Path, mocker: MockerFixture):
    """Test the add_assets command with the --process option for recursive processing."""
    # 1. Setup
    mock_world = mocker.MagicMock(spec=World)

    # Configure the mock to return specific operations
    mock_ingest_op = AssetOperation(
        name="ingest-archive",
        description="",
        add_command_class=IngestArchiveCommand,
    )
    mock_exif_op = AssetOperation(
        name="extract-exif-metadata",
        description="",
        add_command_class=ExtractExifMetadataCommand,
    )

    def get_asset_operation_side_effect(name: str):
        if name == "ingest-archive":
            return mock_ingest_op
        if name == "extract-exif-metadata":
            return mock_exif_op
        return None

    mock_world.get_asset_operation.side_effect = get_asset_operation_side_effect

    mock_file_content = b"This is the content of the new file."

    def mock_stream_factory():
        return io.BytesIO(mock_file_content)

    mock_stream_provider = CallableStreamProvider(mock_stream_factory)

    # Create a side effect function for dispatch_command
    def dispatch_command_side_effect(command: BaseCommand[Any, Any], **_kwargs: Any):
        mock_stream = mocker.AsyncMock()

        if isinstance(command, IngestArchiveCommand):

            async def event_generator(_self: Any):
                yield NewEntityCreatedEvent(entity_id=2, stream_provider=mock_stream_provider, filename="new_file.jpg")

            mock_stream.__aiter__ = event_generator
        elif isinstance(command, ExtractExifMetadataCommand):

            async def event_generator_empty(_self: Any):
                if False:
                    yield

            mock_stream.__aiter__ = event_generator_empty
        elif isinstance(command, FindEntityByFilePropertiesCommand):
            mock_stream.get_one_value.return_value = None
        elif isinstance(command, RegisterLocalFileCommand):
            mock_stream.get_one_value.return_value = 1
        elif isinstance(command, GetMimeTypeCommand):
            if command.entity_id == 1:
                mock_stream.get_one_value.return_value = "application/zip"
            elif command.entity_id == 2:
                mock_stream.get_one_value.return_value = "image/jpeg"
        elif isinstance(command, GetAssetFilenamesCommand):
            if command.entity_id == 1:
                mock_stream.get_one_value.return_value = ["test_archive.zip"]
            elif command.entity_id == 2:
                mock_stream.get_one_value.return_value = ["new_file.jpg"]
        elif isinstance(command, (CheckArchiveCommand, CheckExifMetadataCommand)):
            mock_stream.get_one_value.return_value = False
        else:
            mock_stream.get_all_results.return_value = []

        return mock_stream

    mock_world.dispatch_command.side_effect = dispatch_command_side_effect

    # 2. Create a temporary file
    test_file = tmp_path / "test_archive.zip"
    test_file.write_text("dummy content")

    # 3. Call add_assets
    with mocker.patch("dam_app.cli.assets.get_world", return_value=mock_world):
        await add_assets(
            paths=[test_file],
            recursive=False,
            process=["application/zip:ingest-archive", "image/jpeg:extract-exif-metadata"],
            stop_on_error=False,
        )

    # 4. Assertions
    assert mock_world.dispatch_command.call_count >= 8

    # Find the IngestArchiveCommand and ExtractExifMetadataCommand calls
    ingest_cmd = None
    extract_cmd = None
    for call in mock_world.dispatch_command.call_args_list:
        cmd = call.args[0]
        if isinstance(cmd, IngestArchiveCommand):
            ingest_cmd = cmd
        elif isinstance(cmd, ExtractExifMetadataCommand):
            extract_cmd = cmd

    assert ingest_cmd is not None
    assert ingest_cmd.entity_id == 1
    assert ingest_cmd.stream_provider is None  # Stream is not passed for the initial command

    assert extract_cmd is not None
    assert extract_cmd.entity_id == 2
    assert extract_cmd.stream_provider is not None
    async with extract_cmd.stream_provider.get_stream() as provided_stream:
        assert provided_stream.read() == mock_file_content


@pytest.mark.asyncio
@pytest.mark.usefixtures("capsys")
async def test_add_assets_with_extension_process_option(tmp_path: Path, mocker: MockerFixture):
    """Test the add_assets command with the --process option based on file extension."""
    # 1. Setup
    mock_world = mocker.MagicMock(spec=World)

    # Configure the mock to return a specific operation
    mock_ingest_op = AssetOperation(
        name="ingest-archive",
        description="",
        add_command_class=IngestArchiveCommand,
    )
    mock_world.get_asset_operation.return_value = mock_ingest_op

    # Create a side effect function for dispatch_command
    def dispatch_command_side_effect(command: BaseCommand[Any, Any], **_kwargs: Any):
        mock_stream = mocker.AsyncMock()
        mock_stream.get_all_results.return_value = []
        if isinstance(command, FindEntityByFilePropertiesCommand):
            mock_stream.get_one_value.return_value = None
        elif isinstance(command, RegisterLocalFileCommand):
            mock_stream.get_one_value.return_value = 1
        elif isinstance(command, GetMimeTypeCommand):
            mock_stream.get_one_value.return_value = "application/octet-stream"
        elif isinstance(command, GetAssetFilenamesCommand):
            mock_stream.get_one_value.return_value = ["test_archive.zip"]
        elif isinstance(command, CheckArchiveCommand):
            mock_stream.get_one_value.return_value = False
        else:
            # For AutoSetMimeTypeCommand and IngestArchiveCommand
            async def event_generator_empty(_self: Any):
                if False:
                    yield

            mock_stream.__aiter__ = event_generator_empty

        return mock_stream

    mock_world.dispatch_command.side_effect = dispatch_command_side_effect

    # 2. Create a temporary file
    test_file = tmp_path / "test_archive.zip"
    test_file.write_text("dummy content")

    with mocker.patch("dam_app.cli.assets.get_world", return_value=mock_world):
        # 3. Call add_assets with an extension-based process rule
        await add_assets(
            paths=[test_file],
            recursive=False,
            process=[".zip:ingest-archive"],
            stop_on_error=False,
        )

    # 4. Assertions
    # Verify that IngestArchiveCommand was dispatched
    ingest_cmd_found = False
    for call in mock_world.dispatch_command.call_args_list:
        if isinstance(call.args[0], IngestArchiveCommand):
            ingest_cmd_found = True
            assert call.args[0].entity_id == 1
            break

    assert ingest_cmd_found, "IngestArchiveCommand was not dispatched"


@pytest.mark.asyncio
@pytest.mark.usefixtures("capsys")
async def test_add_assets_with_command_name_process_option(tmp_path: Path, mocker: MockerFixture):
    """Test the add_assets command with the --process option using only the command name."""
    # 1. Setup
    # Reset cache before test to ensure dynamic logic is tested
    ExtractExifMetadataCommand._cached_extensions = None  # type: ignore [protected-access]

    mock_world = mocker.MagicMock(spec=World)

    # Configure the mock to return specific operations
    mock_ingest_op = AssetOperation(
        name="ingest-archive",
        description="",
        add_command_class=IngestArchiveCommand,
    )
    mock_exif_op = AssetOperation(
        name="extract-exif-metadata",
        description="",
        add_command_class=ExtractExifMetadataCommand,
    )

    def get_asset_operation_side_effect(name: str):
        if name == "ingest-archive":
            return mock_ingest_op
        if name == "extract-exif-metadata":
            return mock_exif_op
        return None

    mock_world.get_asset_operation.side_effect = get_asset_operation_side_effect

    # Create a side effect function for dispatch_command
    def dispatch_command_side_effect(command: BaseCommand[Any, Any], **_kwargs: Any):
        mock_stream = mocker.AsyncMock()
        mock_stream.get_all_results.return_value = []
        if isinstance(command, FindEntityByFilePropertiesCommand):
            mock_stream.get_one_value.return_value = None  # File does not exist
        elif isinstance(command, RegisterLocalFileCommand):
            # Return a unique entity ID based on the file path
            if "jpg" in str(command.file_path).lower():
                mock_stream.get_one_value.return_value = 1
            elif "zip" in str(command.file_path):
                mock_stream.get_one_value.return_value = 2
        elif isinstance(command, GetMimeTypeCommand):
            if command.entity_id == 1:
                mock_stream.get_one_value.return_value = None
            elif command.entity_id == 2:
                mock_stream.get_one_value.return_value = "application/zip"
        elif isinstance(command, GetAssetFilenamesCommand):
            if command.entity_id == 1:
                mock_stream.get_one_value.return_value = ["test_image.JPG"]
            elif command.entity_id == 2:
                mock_stream.get_one_value.return_value = ["test_archive.zip"]
        elif isinstance(command, (CheckArchiveCommand, CheckExifMetadataCommand)):
            mock_stream.get_one_value.return_value = False
        else:
            # For other commands like AutoSetMimeType, Ingest, Extract
            async def event_generator_empty(_self: Any):
                if False:
                    yield

            mock_stream.__aiter__ = event_generator_empty
        return mock_stream

    mock_world.dispatch_command.side_effect = dispatch_command_side_effect

    # 2. Create temporary files
    image_file = tmp_path / "test_image.JPG"  # Test case-insensitivity
    image_file.write_text("dummy image")
    archive_file = tmp_path / "test_archive.zip"
    archive_file.write_text("dummy archive")

    # Mock the subprocess call to exiftool
    mock_subprocess_result = mocker.MagicMock()
    mock_subprocess_result.stdout = "Recognized file extensions:\nJPG ZIP"
    with (
        mocker.patch("dam_app.cli.assets.get_world", return_value=mock_world),
        mocker.patch("dam_app.commands.subprocess.run", return_value=mock_subprocess_result),
        mocker.patch("dam_app.commands.shutil.which", return_value="/fake/path/to/exiftool"),
    ):
        # 3. Call add_assets with command name-based process rules
        await add_assets(
            paths=[image_file, archive_file],
            recursive=False,
            process=["extract-exif-metadata", "ingest-archive"],
            stop_on_error=False,
        )

    # 4. Assertions
    extract_calls = [
        call.args[0]
        for call in mock_world.dispatch_command.call_args_list
        if isinstance(call.args[0], ExtractExifMetadataCommand)
    ]
    ingest_calls = [
        call.args[0]
        for call in mock_world.dispatch_command.call_args_list
        if isinstance(call.args[0], IngestArchiveCommand)
    ]

    assert len(extract_calls) >= 1
    assert any(c.entity_id == 1 for c in extract_calls)

    assert len(ingest_calls) >= 1
    assert any(c.entity_id == 2 for c in ingest_calls)


@pytest.mark.asyncio
async def test_process_entities_command(capsys: pytest.CaptureFixture[Any], mocker: MockerFixture):
    """Test the 'process' CLI command."""
    # 1. Setup
    mock_world = mocker.MagicMock(spec=World)
    mock_add_command = mocker.MagicMock()
    mock_operation = AssetOperation(
        name="test-op",
        description="A test operation",
        add_command_class=mock_add_command,  # type: ignore [arg-type]
    )
    mock_world.get_asset_operation.return_value = mock_operation

    # Mock the command dispatcher
    mock_stream = mocker.AsyncMock()
    mock_stream.get_all_results.return_value = []
    mock_world.dispatch_command.return_value = mock_stream

    with mocker.patch("dam_app.cli.assets.get_world", return_value=mock_world):
        # 2. Call the command
        await process_entities(operation_name="test-op", entity_ids=[1, 2])

    # 3. Assertions
    assert mock_world.get_asset_operation.call_count == 1
    assert mock_world.get_asset_operation.call_args.args[0] == "test-op"

    assert mock_add_command.call_count == 2
    mock_add_command.assert_any_call(entity_id=1, stream_provider=None)
    mock_add_command.assert_any_call(entity_id=2, stream_provider=None)

    captured = capsys.readouterr()
    assert "Executing operation 'test-op' on 2 entities..." in captured.out
    assert "Processing entity 1..." in captured.out
    assert "Processing entity 2..." in captured.out
    assert "Processing complete." in captured.out


@pytest.mark.asyncio
async def test_remove_data_command(capsys: pytest.CaptureFixture[Any], mocker: MockerFixture):
    """Test the 'remove-data' CLI command."""
    # 1. Setup
    mock_world = mocker.MagicMock(spec=World)
    mock_remove_command = mocker.MagicMock()
    mock_operation = AssetOperation(
        name="test-op",
        description="A test operation",
        add_command_class=mocker.MagicMock(),  # type: ignore [arg-type]
        remove_command_class=mock_remove_command,  # type: ignore [arg-type]
    )
    mock_world.get_asset_operation.return_value = mock_operation

    # Mock the command dispatcher
    mock_stream = mocker.AsyncMock()
    mock_stream.get_all_results.return_value = []
    mock_world.dispatch_command.return_value = mock_stream

    with mocker.patch("dam_app.cli.assets.get_world", return_value=mock_world):
        # 2. Call the command
        await remove_data(operation_name="test-op", entity_ids=[10])

    # 3. Assertions
    assert mock_world.get_asset_operation.call_count == 1
    mock_remove_command.assert_called_once_with(entity_id=10)
    captured = capsys.readouterr()
    assert "Data removed for entity 10." in captured.out


@pytest.mark.asyncio
async def test_check_data_command(capsys: pytest.CaptureFixture[Any], mocker: MockerFixture):
    """Test the 'check-data' CLI command."""
    # 1. Setup
    mock_world = mocker.MagicMock(spec=World)
    mock_check_command = mocker.MagicMock()
    mock_operation = AssetOperation(
        name="test-op",
        description="A test operation",
        add_command_class=mocker.MagicMock(),  # type: ignore [arg-type]
        check_command_class=mock_check_command,  # type: ignore [arg-type]
    )
    mock_world.get_asset_operation.return_value = mock_operation

    # Mock the command dispatcher to return different results
    async def get_one_value_side_effect():
        return mock_check_command.call_args.kwargs["entity_id"] == 1

    mock_stream = mocker.AsyncMock()
    mock_stream.get_one_value.side_effect = get_one_value_side_effect
    mock_world.dispatch_command.return_value = mock_stream

    with mocker.patch("dam_app.cli.assets.get_world", return_value=mock_world):
        # 2. Call the command
        await check_data(operation_name="test-op", entity_ids=[1, 2])

    # 3. Assertions
    assert mock_check_command.call_count == 2
    captured = capsys.readouterr()
    assert "Entity 1: Check PASSED (True)" in captured.out
    assert "Entity 2: Check FAILED (False)" in captured.out
