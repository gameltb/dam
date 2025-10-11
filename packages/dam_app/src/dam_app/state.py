"""Manages the global, shared state for the CLI application."""

import importlib
import logging
from typing import TYPE_CHECKING

from dam.core.config import Config, WorldDefinition
from dam.core.world import World
from dam.core.world_manager import create_world, world_manager

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
        """Create a new World instance using the core factory and add app-level plugins."""
        if not self.config:
            logger.error("Configuration is not loaded, cannot instantiate world.")
            return None

        world_def = self.get_current_world_def()
        if not world_def:
            logger.error("World definition for '%s' not found.", world_name)
            return None

        logger.info("Lazily instantiating and configuring world: '%s'", world_name)
        try:
            # Use the new core function to create the base world
            world_instance = create_world(world_name, self.config)
        except (ValueError, FileNotFoundError) as e:
            logger.error("Failed to create world '%s'. Reason: %s", world_name, e)
            return None

        # Load all application-level plugins for this world
        world_instance.add_plugin(AppPlugin())
        for plugin_name in world_def.plugins.names:
            plugin = self._load_plugin(plugin_name, world_name)
            if plugin:
                world_instance.add_plugin(plugin)

        # Register the fully configured world
        world_manager.register_world(world_instance)
        return world_instance

    def get_current_world(self) -> World | None:
        """
        Lazily instantiate and return the currently active World object.

        Uses the global `world_manager` to get the world, and if it doesn't
        exist, it instantiates and configures it on-demand.
        """
        if not self.world_name:
            return None

        world = world_manager.get_world(self.world_name)
        if world:
            return world

        return self._instantiate_and_configure_world(self.world_name)


global_state = GlobalState()


def get_world() -> World | None:
    """Get the current world from the global state."""
    return global_state.get_current_world()
