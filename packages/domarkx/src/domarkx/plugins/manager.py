"""Manages the lifecycle of plugins."""

from domarkx.plugins.base import Plugin


class PluginManager:
    """Manages the loading and execution of plugins."""

    def __init__(self) -> None:
        """Initialize the PluginManager."""
        self.plugins: dict[str, Plugin] = {}

    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin."""
        if plugin.type in self.plugins:
            raise ValueError(f"Plugin with type '{plugin.type}' is already registered.")
        self.plugins[plugin.type] = plugin

    def get_plugin(self, plugin_type: str) -> Plugin | None:
        """Get a plugin by its type."""
        return self.plugins.get(plugin_type)
