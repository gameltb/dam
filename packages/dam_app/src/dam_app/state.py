"""Manages the global, shared state for the CLI application."""

import importlib
import logging

from dam.core.config import WorldConfig
from dam.core.world import World

from dam_app.config import Config, WorldDefinition
from dam_app.plugin import AppPlugin

logger = logging.getLogger(__name__)

# A mapping from plugin package names to their main Plugin class names.
PLUGIN_CLASS_MAP = {
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

    def __init__(self):
        """Initialize the GlobalState."""
        self.config: Config | None = None
        self.world_name: str | None = None
        self._worlds_cache: dict[str, World] = {}

    def get_current_world_def(self) -> WorldDefinition | None:
        """Retrieve the configuration definition for the currently active world."""
        if self.config and self.world_name:
            return self.config.worlds.get(self.world_name)
        return None

    def get_current_world(self) -> World | None:
        """
        Lazily instantiate and return the currently active World object.

        Caches the instantiated world for subsequent calls.
        """
        if not self.world_name:
            return None

        # Return from cache if already instantiated
        if self.world_name in self._worlds_cache:
            return self._worlds_cache[self.world_name]

        # Instantiate on-demand
        world_def = self.get_current_world_def()
        if not world_def:
            return None

        logger.info("Lazily instantiating world: '%s'", self.world_name)
        world_config = WorldConfig(
            name=self.world_name,
            DATABASE_URL=world_def.db.url,
            plugin_settings=world_def.model_dump().get("plugin_settings", {}),
        )
        world_instance = World(world_config=world_config)

        # Load all configured plugins for this world
        world_instance.add_plugin(AppPlugin())
        for plugin_name in world_def.plugins.names:
            if plugin_name not in PLUGIN_CLASS_MAP:
                logger.warning("Unknown plugin '%s' for world '%s'. Skipping.", plugin_name, self.world_name)
                continue
            try:
                module_path = plugin_name.replace("-", "_")
                class_name = PLUGIN_CLASS_MAP[plugin_name]
                plugin_module = importlib.import_module(f"{module_path}.plugin")
                plugin_class = getattr(plugin_module, class_name)
                world_instance.add_plugin(plugin_class())
                logger.info("Loaded plugin '%s' for world '%s'.", plugin_name, self.world_name)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    "Could not load plugin '%s' for world '%s'. Reason: %s.", plugin_name, self.world_name, e
                )

        # Cache and return the new instance
        self._worlds_cache[self.world_name] = world_instance
        return world_instance


global_state = GlobalState()


def get_world() -> World | None:
    """Get the current world from the global state."""
    return global_state.get_current_world()
