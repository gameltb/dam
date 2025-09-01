import pytest_asyncio
from dam.core.world import World
from dam_app import AppPlugin
from dam_fs import FsPlugin
from dam_media_audio import AudioPlugin
from dam_semantic import SemanticPlugin
from dam_test_utils.fixtures import _setup_world, _teardown_world_async

pytest_plugins = ["dam_test_utils.fixtures"]


@pytest_asyncio.fixture(scope="function")
async def test_world_alpha(settings_override) -> World:
    """Override the vanilla test_world_alpha to include plugins."""
    plugins = [AppPlugin(), FsPlugin(), SemanticPlugin(), AudioPlugin()]
    world = await _setup_world("test_world_alpha", settings_override, plugins=plugins)
    yield world
    await _teardown_world_async(world)
