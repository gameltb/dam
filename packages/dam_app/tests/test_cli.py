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
    assert "Running ExtractMetadataCommand on entity 1 at depth 0" in captured.out


@pytest.mark.asyncio
async def test_add_assets_with_recursive_process_option(capsys: CaptureFixture[Any], tmp_path: Path):
    """Test the add_assets command with the --process option for recursive processing."""
    from dam.models.conceptual.mime_type_concept_component import MimeTypeConceptComponent
    from dam.models.metadata.content_mime_type_component import ContentMimeTypeComponent
    from dam.system_events import NewEntityCreatedEvent
    from dam_archive.commands import IngestArchiveMembersCommand
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

        if isinstance(command, IngestArchiveMembersCommand):

            async def event_generator(self: AsyncMock):
                yield NewEntityCreatedEvent(entity_id=2, depth=1, file_stream=None)

            mock_stream.__aiter__ = event_generator
            mock_stream.get_one_value.return_value = None

        elif isinstance(command, ExtractMetadataCommand):

            async def event_generator_empty(self: AsyncMock):
                if False:
                    yield

            mock_stream.__aiter__ = event_generator_empty
            mock_stream.get_one_value.return_value = None

        elif isinstance(command, FindEntityByFilePropertiesCommand):
            mock_stream.get_one_value.return_value = None
        elif isinstance(command, RegisterLocalFileCommand):
            mock_stream.get_one_value.return_value = 1
        else:
            # For AutoSetMimeTypeCommand
            mock_stream.get_all_results.return_value = []

        return mock_stream

    mock_world.dispatch_command.side_effect = dispatch_command_side_effect

    # Mock get_component to return different MIME types
    archive_mime_concept = MimeTypeConceptComponent(
        concept_name="application/zip", concept_description=None, mime_type="application/zip"
    )
    archive_mime_component = ContentMimeTypeComponent(mime_type_concept_id=2)
    archive_mime_component.mime_type_concept = archive_mime_concept

    image_mime_concept = MimeTypeConceptComponent(
        concept_name="image/jpeg", concept_description=None, mime_type="image/jpeg"
    )
    image_mime_component = ContentMimeTypeComponent(mime_type_concept_id=3)
    image_mime_component.mime_type_concept = image_mime_concept

    from dam_fs.models import FilenameComponent

    async def get_component_side_effect(session: Any, entity_id: int, component_type: Any):
        if component_type == ContentMimeTypeComponent:
            if entity_id == 1:
                return archive_mime_component
            elif entity_id == 2:
                return image_mime_component
        elif component_type == FilenameComponent:
            if entity_id == 1:
                return FilenameComponent(filename="test_archive.zip")
        return None

    # 2. Create a temporary file
    test_file = tmp_path / "test_archive.zip"
    test_file.write_text("dummy content")

    # 3. Call add_assets
    with (
        patch("dam_app.cli.assets.get_world", return_value=mock_world),
        patch("dam_app.cli.assets.dam_ecs_functions.get_component") as mock_get_component,
    ):
        mock_get_component.side_effect = get_component_side_effect
        await add_assets(
            paths=[test_file],
            recursive=False,
            process=[
                "application/zip:IngestArchiveMembersCommand",
                "image/jpeg:ExtractMetadataCommand",
            ],
        )

    # 4. Assertions
    assert mock_world.dispatch_command.call_count == 6
    ingest_cmd = mock_world.dispatch_command.call_args_list[3].args[0]
    assert isinstance(ingest_cmd, IngestArchiveMembersCommand)
    assert ingest_cmd.entity_id == 1
    assert ingest_cmd.depth == 0

    extract_cmd = mock_world.dispatch_command.call_args_list[5].args[0]
    assert isinstance(extract_cmd, ExtractMetadataCommand)
    assert extract_cmd.entity_id == 2
    assert extract_cmd.depth == 1


@pytest.mark.asyncio
async def test_add_assets_with_extension_process_option(capsys: CaptureFixture[Any], tmp_path: Path):
    """Test the add_assets command with the --process option based on file extension."""
    from dam.models.conceptual.mime_type_concept_component import MimeTypeConceptComponent
    from dam.models.metadata.content_mime_type_component import ContentMimeTypeComponent
    from dam_archive.commands import IngestArchiveMembersCommand
    from dam_fs.commands import FindEntityByFilePropertiesCommand, RegisterLocalFileCommand
    from dam_fs.models import FilenameComponent

    from dam_app.cli.assets import add_assets

    # 1. Setup
    mock_world = MagicMock(spec=World)
    mock_session = AsyncMock()
    mock_world.db_session_maker.return_value.__aenter__.return_value = mock_session

    # Create a side effect function for dispatch_command
    def dispatch_command_side_effect(command: BaseCommand[Any]):
        mock_stream = AsyncMock()
        mock_stream.get_all_results.return_value = []
        if isinstance(command, FindEntityByFilePropertiesCommand):
            mock_stream.get_one_value.return_value = None
        elif isinstance(command, RegisterLocalFileCommand):
            mock_stream.get_one_value.return_value = 1
        else:
            # For AutoSetMimeTypeCommand and IngestArchiveMembersCommand
            async def event_generator_empty(self: AsyncMock):
                if False:
                    yield

            mock_stream.__aiter__ = event_generator_empty

        return mock_stream

    mock_world.dispatch_command.side_effect = dispatch_command_side_effect

    # Mock get_component
    mock_mime_type_concept = MimeTypeConceptComponent(
        concept_name="application/octet-stream", concept_description=None, mime_type="application/octet-stream"
    )
    mock_mime_type_component = ContentMimeTypeComponent(mime_type_concept_id=1)
    mock_mime_type_component.mime_type_concept = mock_mime_type_concept

    async def get_component_side_effect_ext(session: Any, entity_id: int, component_type: Any):
        if component_type == ContentMimeTypeComponent:
            return mock_mime_type_component
        elif component_type == FilenameComponent:
            return FilenameComponent(filename="test_archive.zip")
        return None

    # 2. Create a temporary file
    test_file = tmp_path / "test_archive.zip"
    test_file.write_text("dummy content")

    with (
        patch("dam_app.cli.assets.get_world", return_value=mock_world),
        patch("dam_app.cli.assets.dam_ecs_functions.get_component", side_effect=get_component_side_effect_ext),
    ):
        # 3. Call add_assets with an extension-based process rule
        await add_assets(
            paths=[test_file],
            recursive=False,
            process=[".zip:IngestArchiveMembersCommand"],
        )

    # 4. Assertions
    # Verify that IngestArchiveMembersCommand was dispatched
    ingest_cmd_found = False
    for call in mock_world.dispatch_command.call_args_list:
        if isinstance(call.args[0], IngestArchiveMembersCommand):
            ingest_cmd_found = True
            assert call.args[0].entity_id == 1
            break

    assert ingest_cmd_found, "IngestArchiveMembersCommand was not dispatched"
