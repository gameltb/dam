"""Utility functions for discovering and loading plugins."""

import importlib.metadata
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dam.core.plugin import Plugin

logger = logging.getLogger(__name__)


def load_plugin(plugin_name: str) -> "Plugin | None":
    """Dynamically load a plugin instance by its package name."""
    try:
        eps = importlib.metadata.entry_points(group="dam.plugins", name=plugin_name)
        try:
            ep = next(iter(eps))
        except StopIteration:
            logger.warning("Unknown plugin '%s' not found.", plugin_name)
            return None

        plugin_class = ep.load()
        logger.info("Loaded plugin '%s'.", plugin_name)
        return plugin_class()
    except Exception as e:
        logger.warning("Could not load plugin '%s'. Reason: %s.", plugin_name, e)
        return None


def get_all_plugins(
    enabled_plugins: list[str] | None = None,
) -> dict[str, "Plugin"]:
    """
    Load and return a dictionary of plugins.

    If `enabled_plugins` is provided, only plugins from that list will be loaded.
    Otherwise, all discoverable plugins will be loaded.
    """
    plugins: dict[str, Plugin] = {}
    try:
        entry_points = importlib.metadata.entry_points(group="dam.plugins")
        for ep in entry_points:
            # If a whitelist is provided, only load plugins that are in it.
            if enabled_plugins is not None and ep.name not in enabled_plugins:
                continue

            try:
                plugin_class = ep.load()
                plugin_instance = plugin_class()
                plugins[ep.name] = plugin_instance
                logger.info("Loaded plugin '%s'.", ep.name)
            except Exception as e:
                logger.warning("Could not load plugin '%s'. Reason: %s.", ep.name, e)
    except Exception as e:
        logger.error("Error discovering plugins for dam.plugins group: %s", e)

    return plugins
