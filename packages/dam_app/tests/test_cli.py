from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dam.core.config import Settings
from dam.core.world import World
from pytest import CaptureFixture

from dam_app.cli.assets import clear_archive_info


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

    with patch("dam_app.cli.assets.get_world", return_value=mock_world):
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
