"""Fixtures for the `dam_fs` package tests."""

import pytest
from dam.core.world import World

from dam_fs.plugin import FsPlugin

pytest_plugins = ["dam_test_utils.fixtures"]


@pytest.fixture(autouse=True)
def setup_world_with_plugins(test_world_alpha: World) -> None:
    """Set up the world with the FsPlugin."""
    test_world_alpha.add_plugin(FsPlugin())
