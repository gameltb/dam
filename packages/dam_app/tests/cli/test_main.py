"""Tests for the DAM application's CLI commands."""

from typing import Any

import pytest

from dam_app.main import cli_list_worlds
from dam_app.state import global_state


def test_cli_list_worlds(capsys: pytest.CaptureFixture[Any]):
    """Test the list-worlds command."""
    # 1. Setup: Manually populate the global state to simulate loaded components.
    global_state.loaded_components = {
        "test_world_alpha": {},  # The values can be empty dicts for this test
        "test_world_beta": {},
    }
    global_state.world_name = "test_world_alpha"  # Set an active world

    # 2. Execute
    cli_list_worlds()

    # 3. Assert
    captured = capsys.readouterr()
    assert "test_world_alpha (active)" in captured.out
    assert "test_world_beta" in captured.out

    # 4. Teardown
    global_state.loaded_components = {}
    global_state.world_name = None
