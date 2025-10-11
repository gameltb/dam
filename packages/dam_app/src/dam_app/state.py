"""Manages the global, shared state for the CLI application."""

import importlib
import logging
from typing import TYPE_CHECKING

from dam import world_manager
from dam.core.config import WorldConfig
from dam.core.world import World

from dam_app.config import Config, WorldDefinition
from dam_app.plugin import AppPlugin

if TYPE_CHECKING:
    from dam.core.plugin import Plugin

logger = logging.getLogger(__name__)

# A mapping from plugin package names to their main Plugin class names.
PLUGIN_CLASS_MAP: dict[str, str] = {
    "dam-fs": "FsPlugin",
    "dam-source": "SourcePlugin",
    "dam-archive": "ArchivePlugin",
    "dam-media-image": "ImagePlugin",
    "dam-media-audio": "AudioPlugin",
    "dam-media-transcode": "TranscodePlugin",
    "dam-psp": "PspPlugin",
    "dam-semantic": "SemanticPlugin",
    "dam-sire": "SirePlugin",
}


class GlobalState:
    """A singleton class to hold the global state of the CLI application."""

    def __init__(self) -> None:
        """Initialize the GlobalState."""
        self.config: Config | None = None
        self.world_name: str | None = None
        self._instantiated_worlds: set[str] = set()

    def get_current_world_def(self) -> WorldDefinition | None:
        """Retrieve the configuration definition for the currently active world."""
        if self.config and self.world_name:
            return self.config.worlds.get(self.world_name)
        return None

    def _load_plugin(self, plugin_name: str, world_name: str) -> "Plugin | None":
        """Dynamically load a plugin class by its package name."""
        if plugin_name not in PLUGIN_CLASS_MAP:
            logger.warning("Unknown plugin '%s' for world '%s'. Skipping.", plugin_name, world_name)
            return None
        try:
            module_path = plugin_name.replace("-", "_")
            class_name = PLUGIN_CLASS_MAP[plugin_name]
            plugin_module = importlib.import_module(f"{module_path}.plugin")
            plugin_class: type[Plugin] = getattr(plugin_module, class_name)
            logger.info("Loaded plugin '%s' for world '%s'.", plugin_name, world_name)
            return plugin_class()
        except (ImportError, AttributeError) as e:
            logger.warning("Could not load plugin '%s' for world '%s'. Reason: %s.", plugin_name, world_name, e)
            return None

    def _instantiate_and_configure_world(self, world_name: str) -> World | None:
        """Create a new World instance and configure it with plugins."""
        world_def = self.get_current_world_def()
        if not world_def:
            logger.error("World definition for '%s' not found.", world_name)
            return None

        logger.info("Lazily instantiating and configuring world: '%s'", world_name)
        world_config = WorldConfig(
            name=world_name,
            DATABASE_URL=world_def.db.url,
            plugin_settings=world_def.model_dump(mode="python").get("plugin_settings", {}),
        )
        world_instance = World(world_config=world_config)

        # Load all configured plugins for this world
        world_instance.add_plugin(AppPlugin())
        for plugin_name in world_def.plugins.names:
            plugin = self._load_plugin(plugin_name, world_name)
            if plugin:
                world_instance.add_plugin(plugin)

        world_manager.register_world(world_instance)
        self._instantiated_worlds.add(world_name)
        return world_instance

    def get_current_world(self) -> World | None:
        """
        Lazily instantiate and return the currently active World object.

        Uses the global `world_manager` to get the world, and if it doesn't
        exist, it instantiates and configures it on-demand.
        """
        if not self.world_name:
            return None

        # Check if the world is already registered in the central manager
        world = world_manager.get_world(self.world_name)
        if world:
            return world

        # If not, instantiate it, which also registers it.
        return self._instantiate_and_configure_world(self.world_name)


global_state = GlobalState()


def get_world() -> World | None:
    """Get the current world from the global state."""
    return global_state.get_current_world()
