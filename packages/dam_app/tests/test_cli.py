from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dam.commands.core import BaseCommand
from dam.core.config import Settings
from dam.core.world import World
from pytest import CaptureFixture

from dam_app.cli.archive import clear_archive_info


@pytest.mark.serial
def test_cli_list_worlds(settings_override: Settings, capsys: CaptureFixture[Any]):
    """Test the list-worlds command."""
    from dam_app.main import cli_list_worlds, create_and_register_all_worlds_from_settings

    # Ensure worlds are registered
    create_and_register_all_worlds_from_settings(app_settings=settings_override)

    cli_list_worlds()

    captured = capsys.readouterr()
    assert "test_world_alpha" in captured.out
    assert "test_world_beta" in captured.out


@pytest.mark.asyncio
async def test_clear_archive_info_command(capsys: CaptureFixture[Any]):
    """Test the clear-archive-info CLI command by calling the function directly."""
    entity_id = 123
    mock_world = MagicMock(spec=World)
    mock_stream = AsyncMock()
    mock_stream.get_all_results.return_value = []
    mock_world.dispatch_command.return_value = mock_stream

    with patch("dam_app.cli.archive.get_world", return_value=mock_world):
        await clear_archive_info(entity_id=entity_id)

        # Verify that the correct command was dispatched
        mock_world.dispatch_command.assert_called_once()
        dispatched_command = mock_world.dispatch_command.call_args[0][0]
        from dam_archive.commands import ClearArchiveComponentsCommand

        assert isinstance(dispatched_command, ClearArchiveComponentsCommand)
        assert dispatched_command.entity_id == entity_id

    captured = capsys.readouterr()
    assert f"Clearing archive info for entity: {entity_id}" in captured.out
    assert "Archive info clearing process complete." in captured.out


@pytest.mark.asyncio
async def test_add_assets_with_recursive_process_option(capsys: CaptureFixture[Any], tmp_path: Path):
    """Test the add_assets command with the --process option for recursive processing."""
    import io

    from dam.commands.asset_commands import GetAssetFilenamesCommand, GetMimeTypeCommand
    from dam.system_events.entity_events import NewEntityCreatedEvent
    from dam_archive.commands import IngestArchiveCommand
    from dam_fs.commands import FindEntityByFilePropertiesCommand, RegisterLocalFileCommand

    from dam_app.cli.assets import add_assets
    from dam_app.commands import ExtractExifMetadataCommand

    # 1. Setup
    mock_world = MagicMock(spec=World)
    mock_session = AsyncMock()
    mock_world.db_session_maker.return_value.__aenter__.return_value = mock_session

    mock_file_content = b"This is the content of the new file."

    def mock_stream_provider():
        return io.BytesIO(mock_file_content)

    # Create a side effect function for dispatch_command
    def dispatch_command_side_effect(command: BaseCommand[Any, Any], **kwargs: Any):
        mock_stream = AsyncMock()

        if isinstance(command, IngestArchiveCommand):

            async def event_generator(self: AsyncMock):
                yield NewEntityCreatedEvent(entity_id=2, stream_provider=mock_stream_provider, filename="new_file.jpg")

            mock_stream.__aiter__ = event_generator
        elif isinstance(command, ExtractExifMetadataCommand):

            async def event_generator_empty(self: AsyncMock):
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
        else:
            mock_stream.get_all_results.return_value = []

        return mock_stream

    mock_world.dispatch_command.side_effect = dispatch_command_side_effect

    # 2. Create a temporary file
    test_file = tmp_path / "test_archive.zip"
    test_file.write_text("dummy content")

    # 3. Call add_assets
    with patch("dam_app.cli.assets.get_world", return_value=mock_world):
        await add_assets(
            paths=[test_file],
            recursive=False,
            process=["application/zip:IngestArchiveCommand", "image/jpeg:ExtractExifMetadataCommand"],
            stop_on_error=False,
        )

    # 4. Assertions
    assert mock_world.dispatch_command.call_count == 8

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
    provided_stream = extract_cmd.stream_provider()
    assert provided_stream.read() == mock_file_content


@pytest.mark.asyncio
async def test_add_assets_with_extension_process_option(capsys: CaptureFixture[Any], tmp_path: Path):
    """Test the add_assets command with the --process option based on file extension."""
    from dam.commands.asset_commands import GetAssetFilenamesCommand, GetMimeTypeCommand
    from dam_archive.commands import IngestArchiveCommand
    from dam_fs.commands import FindEntityByFilePropertiesCommand, RegisterLocalFileCommand

    from dam_app.cli.assets import add_assets

    # 1. Setup
    mock_world = MagicMock(spec=World)
    mock_session = AsyncMock()
    mock_world.db_session_maker.return_value.__aenter__.return_value = mock_session

    # Create a side effect function for dispatch_command
    def dispatch_command_side_effect(command: BaseCommand[Any, Any], **kwargs: Any):
        mock_stream = AsyncMock()
        mock_stream.get_all_results.return_value = []
        if isinstance(command, FindEntityByFilePropertiesCommand):
            mock_stream.get_one_value.return_value = None
        elif isinstance(command, RegisterLocalFileCommand):
            mock_stream.get_one_value.return_value = 1
        elif isinstance(command, GetMimeTypeCommand):
            mock_stream.get_one_value.return_value = "application/octet-stream"
        elif isinstance(command, GetAssetFilenamesCommand):
            mock_stream.get_one_value.return_value = ["test_archive.zip"]
        else:
            # For AutoSetMimeTypeCommand and IngestArchiveCommand
            async def event_generator_empty(self: AsyncMock):
                if False:
                    yield

            mock_stream.__aiter__ = event_generator_empty

        return mock_stream

    mock_world.dispatch_command.side_effect = dispatch_command_side_effect

    # 2. Create a temporary file
    test_file = tmp_path / "test_archive.zip"
    test_file.write_text("dummy content")

    with patch("dam_app.cli.assets.get_world", return_value=mock_world):
        # 3. Call add_assets with an extension-based process rule
        await add_assets(
            paths=[test_file],
            recursive=False,
            process=[".zip:IngestArchiveCommand"],
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
async def test_add_assets_with_command_name_process_option(capsys: CaptureFixture[Any], tmp_path: Path):
    """Test the add_assets command with the --process option using only the command name."""
    from dam.commands.asset_commands import GetAssetFilenamesCommand, GetMimeTypeCommand
    from dam_archive.commands import IngestArchiveCommand
    from dam_fs.commands import FindEntityByFilePropertiesCommand, RegisterLocalFileCommand

    from dam_app.cli.assets import add_assets
    from dam_app.commands import ExtractExifMetadataCommand

    # 1. Setup
    # Reset cache before test to ensure dynamic logic is tested
    ExtractExifMetadataCommand._cached_extensions = None  # type: ignore [protected-access]

    mock_world = MagicMock(spec=World)
    mock_session = AsyncMock()
    mock_world.db_session_maker.return_value.__aenter__.return_value = mock_session

    # Create a side effect function for dispatch_command
    def dispatch_command_side_effect(command: BaseCommand[Any, Any], **kwargs: Any):
        mock_stream = AsyncMock()
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
        else:
            # For other commands like AutoSetMimeType, Ingest, Extract
            async def event_generator_empty(self: AsyncMock):
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
    mock_subprocess_result = MagicMock()
    mock_subprocess_result.stdout = "Recognized file extensions:\nJPG ZIP"
    with (
        patch("dam_app.cli.assets.get_world", return_value=mock_world),
        patch("dam_app.commands.subprocess.run", return_value=mock_subprocess_result),
        patch("dam_app.commands.shutil.which", return_value="/fake/path/to/exiftool"),
    ):
        # 3. Call add_assets with command name-based process rules
        await add_assets(
            paths=[image_file, archive_file],
            recursive=False,
            process=["ExtractExifMetadataCommand", "IngestArchiveCommand"],
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

    assert len(extract_calls) == 2
    assert any(c.entity_id == 1 for c in extract_calls)
    assert any(c.entity_id == 2 for c in extract_calls)

    assert len(ingest_calls) == 1
    assert ingest_calls[0].entity_id == 2
