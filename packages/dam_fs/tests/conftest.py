import pytest

pytest_plugins = ["dam_test_utils.fixtures"]

@pytest.fixture(autouse=True)
def setup_world_with_plugins(test_world_alpha):
    from dam_media_audio.plugin import AudioPlugin
    test_world_alpha.add_plugin(AudioPlugin())
