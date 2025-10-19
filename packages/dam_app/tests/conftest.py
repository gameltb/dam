"""Pytest configuration and fixtures for the DAM application tests."""

pytest_plugins = ["dam_test_utils.fixtures"]


# @pytest.fixture(autouse=True)
# def setup_world_with_plugins(test_world_alpha: World):
#     """Automatically set up the test world with all necessary plugins."""
#     test_world_alpha.add_plugin(FsPlugin())
#     test_world_alpha.add_plugin(AppPlugin())
#     test_world_alpha.add_plugin(ArchivePlugin())
