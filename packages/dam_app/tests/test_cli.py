from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dam.core.commands import BaseCommand
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
async def test_add_assets_with_process_option(capsys: CaptureFixture[Any], tmp_path: Path):
    """Test the add_assets command with the --process option."""
    from dam.commands.asset_commands import AutoSetMimeTypeCommand
    from dam.models.conceptual.mime_type_concept_component import MimeTypeConceptComponent
    from dam.models.metadata.content_mime_type_component import ContentMimeTypeComponent
    from dam_fs.commands import FindEntityByFilePropertiesCommand, RegisterLocalFileCommand

    from dam_app.cli.assets import add_assets
    from dam_app.commands import ExtractMetadataCommand

    # 1. Setup
    mock_world = MagicMock(spec=World)
    mock_session = AsyncMock()
    mock_world.db_session_maker.return_value.__aenter__.return_value = mock_session

    # Create a side effect function for dispatch_command
    def dispatch_command_side_effect(command: BaseCommand[Any]):
        mock_stream = AsyncMock()
        mock_stream.get_all_results.return_value = []
        if isinstance(command, FindEntityByFilePropertiesCommand):
            mock_stream.get_one_value.return_value = None  # File does not exist
        elif isinstance(command, RegisterLocalFileCommand):
            mock_stream.get_one_value.return_value = 1  # Return new entity ID
        else:
            mock_stream.get_one_value.return_value = None
        return mock_stream

    mock_world.dispatch_command.side_effect = dispatch_command_side_effect

    # Mock get_component to return a ContentMimeTypeComponent
    mock_mime_type_concept = MimeTypeConceptComponent(
        concept_name="image/jpeg", concept_description=None, mime_type="image/jpeg"
    )
    mock_mime_type_component = ContentMimeTypeComponent(mime_type_concept_id=1)
    mock_mime_type_component.mime_type_concept = mock_mime_type_concept

    # 2. Create a temporary file
    test_file = tmp_path / "test_image.jpg"
    test_file.write_text("dummy content")

    # We need to patch the ecs_functions.get_component, not a method on the world
    with (
        patch("dam_app.cli.assets.get_world", return_value=mock_world),
        patch(
            "dam_app.cli.assets.dam_ecs_functions.get_component", return_value=mock_mime_type_component
        ) as mock_get_component,
    ):
        # 3. Call add_assets
        await add_assets(
            paths=[test_file],
            recursive=False,
            process=["image/jpeg:ExtractMetadataCommand"],
        )

    # 4. Assertions
    # Verify that the correct commands were dispatched
    assert mock_world.dispatch_command.call_count == 4

    # Call 1: FindEntityByFilePropertiesCommand
    find_cmd = mock_world.dispatch_command.call_args_list[0][0][0]
    assert isinstance(find_cmd, FindEntityByFilePropertiesCommand)

    # Call 2: RegisterLocalFileCommand
    register_cmd = mock_world.dispatch_command.call_args_list[1][0][0]
    assert isinstance(register_cmd, RegisterLocalFileCommand)
    assert register_cmd.file_path == test_file

    # Call 3: AutoSetMimeTypeCommand
    auto_set_mime_cmd = mock_world.dispatch_command.call_args_list[2][0][0]
    assert isinstance(auto_set_mime_cmd, AutoSetMimeTypeCommand)
    assert auto_set_mime_cmd.entity_id == 1

    # Assert that get_component was called correctly
    mock_get_component.assert_called_once_with(mock_session, 1, ContentMimeTypeComponent)

    # Call 4: ExtractMetadataCommand
    extract_cmd = mock_world.dispatch_command.call_args_list[3][0][0]
    assert isinstance(extract_cmd, ExtractMetadataCommand)
    assert extract_cmd.entity_id == 1

    captured = capsys.readouterr()
    assert "Processed test_image.jpg with ExtractMetadataCommand" in captured.out
