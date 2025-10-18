"""Manages the global, shared state for the CLI application."""

import logging
from typing import TYPE_CHECKING

from dam import world_manager
from dam.core import plugin_loader
from dam.core.world import World
from dam.models.config import ConfigComponent

from dam_app.plugin import AppPlugin

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class GlobalState:
    """A singleton class to hold the global state of the CLI application."""

    def __init__(self) -> None:
        """Initialize the GlobalState."""
        self.loaded_components: dict[str, dict[str, ConfigComponent]] | None = None
        self.world_name: str | None = None
        self._instantiated_worlds: set[str] = set()

    def _instantiate_and_configure_world(self, world_name: str) -> World | None:
        """Create a new World instance and configure it with plugins and settings."""
        if not self.loaded_components or world_name not in self.loaded_components:
            logger.error("Configuration components for world '%s' not loaded.", world_name)
            return None

        logger.info("Lazily instantiating and configuring world: '%s'", world_name)

        # Create a world instance without the old config object.
        world_instance = World(name=world_name)

        # Get the loaded components for the current world.
        world_settings = self.loaded_components[world_name]

        # Inject each loaded ConfigComponent as a resource in the world.
        for component in world_settings.values():
            world_instance.add_resource(component, component.__class__)
        logger.debug("Injected all settings components as resources for world '%s'.", world_name)

        # Load all configured plugins for this world. The list of plugins is
        # determined by the keys of the loaded settings dictionary.
        world_instance.add_plugin(AppPlugin())
        for plugin_name in world_settings:
            plugin = plugin_loader.load_plugin(plugin_name)
            if plugin:
                world_instance.add_plugin(plugin)

        world_manager.register_world(world_instance)
        self._instantiated_worlds.add(world_name)
        return world_instance

    def get_current_world(self) -> World | None:
        """
        Lazily instantiate and return the currently active World object.

        Uses the global `world_manager` to get the world, and if it doesn't
        exist, it instantiates and configures it on-demand using the pre-loaded
        configuration components.
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
