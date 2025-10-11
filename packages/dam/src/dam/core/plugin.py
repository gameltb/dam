"""Core plugin protocol for the DAM system."""

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol

from dam.core.config import PluginSettings

if TYPE_CHECKING:
    from dam.core.world import World

logger = logging.getLogger(__name__)


class Plugin(Protocol):
    """A protocol for plugins that can be added to a World."""

    def build(self, world: "World") -> None:
        """
        Build the plugin, adding systems, resources, etc. to the world.

        Args:
            world: The world to build the plugin into.

        """
        ...


def discover_plugin_settings() -> dict[str, type[PluginSettings]]:
    """
    Discover all registered plugin settings classes via entry points.

    Scans for packages that have registered an entry point in the
    `dam.plugin_settings` group.

    Returns:
        A dictionary mapping the plugin name to its settings class.

    """
    settings_map: dict[str, type[PluginSettings]] = {}
    try:
        eps = entry_points(group="dam.plugin_settings")
    except Exception:
        # Handle cases where entry points might not be available or error out,
        # especially in certain testing or packaging environments.
        logger.warning("Could not discover plugin settings via entry points.", exc_info=True)
        return {}

    for ep in eps:
        try:
            settings_class = ep.load()
            if not issubclass(settings_class, PluginSettings):
                logger.warning(
                    "Discovered plugin settings class '%s' from entry point '%s' is not a subclass of PluginSettings. Skipping.",
                    settings_class.__name__,
                    ep.name,
                )
                continue
            if ep.name in settings_map:
                logger.warning(
                    "Duplicate plugin settings entry point name '%s'. The later one will be used.",
                    ep.name,
                )
            settings_map[ep.name] = settings_class
            logger.info("Discovered and loaded plugin settings for '%s'.", ep.name)
        except Exception:
            logger.warning("Failed to load plugin settings from entry point '%s'.", ep.name, exc_info=True)

    return settings_map
