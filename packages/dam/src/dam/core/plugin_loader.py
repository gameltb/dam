"""Utility functions for discovering and loading plugins."""

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dam.core.plugin import Plugin

logger = logging.getLogger(__name__)

# A mapping from plugin package names to their main Plugin class names.
# This will eventually be replaced by an automatic discovery mechanism.
PLUGIN_CLASS_MAP: dict[str, str] = {
    "core": "CorePlugin",  # Add core plugin here
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


def load_plugin(plugin_name: str) -> "Plugin | None":
    """Dynamically load a plugin instance by its package name."""
    if plugin_name not in PLUGIN_CLASS_MAP:
        logger.warning("Unknown plugin '%s' defined in config. Skipping.", plugin_name)
        return None
    try:
        module_path = "dam.plugins.core" if plugin_name == "core" else plugin_name.replace("-", "_") + ".plugin"

        class_name = PLUGIN_CLASS_MAP[plugin_name]
        plugin_module = importlib.import_module(module_path)
        plugin_class: type[Plugin] = getattr(plugin_module, class_name)
        logger.info("Loaded plugin '%s'.", plugin_name)
        return plugin_class()
    except (ImportError, AttributeError) as e:
        logger.warning("Could not load plugin '%s'. Reason: %s.", plugin_name, e)
        return None


def get_all_plugins() -> dict[str, "Plugin"]:
    """Load and return all known plugins."""
    plugins: dict[str, Plugin] = {}
    for name in PLUGIN_CLASS_MAP:
        plugin = load_plugin(name)
        if plugin:
            plugins[name] = plugin
    return plugins
