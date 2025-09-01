import pytest

from dam_fs.plugin import FsPlugin

pytest_plugins = ["dam_test_utils.fixtures"]


@pytest.fixture(autouse=True)
def setup_world_with_plugins(test_world_alpha):
    test_world_alpha.add_plugin(FsPlugin())
