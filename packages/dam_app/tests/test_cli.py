from typing import Any

import pytest
from dam.core.config import Settings
from pytest import CaptureFixture


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
