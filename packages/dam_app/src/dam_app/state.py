"""Manages the global, shared state for the CLI application."""

import logging
from typing import TYPE_CHECKING

import tomli
from dam import world_manager
from dam.core.config_loader import TOMLConfig, find_config_file
from dam.core.world import World
from dam.core.world_manager import create_world_from_components
from dam.models.config import ConfigComponent

from dam_app.plugin import AppPlugin

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class GlobalState:
    """A singleton class to hold the global state of the CLI application."""

    def __init__(self) -> None:
        """Initialize the GlobalState."""
        self.world_name: str | None = None
        # Note: _instantiated_worlds is removed as world_manager now tracks this.

    def get_current_world(self) -> World | None:
        """
        Lazily instantiate and return the currently active World object.

        If the world is not already registered, this function will find the dam.toml,
        parse the configuration for the current world, and build it on-demand.
        """
        if not self.world_name:
            return None

        # Check if the world is already registered in the central manager
        world = world_manager.get_world(self.world_name)
        if world:
            return world

        # --- On-demand world instantiation from dam.toml ---
        logger.info("Lazily instantiating world '%s' from configuration file.", self.world_name)
        try:
            config_path = find_config_file()
            if not config_path:
                raise FileNotFoundError("Configuration file 'dam.toml' not found.")

            with config_path.open("rb") as f:
                toml_data = tomli.load(f)

            # Validate the full TOML structure to get our world's definition
            toml_config = TOMLConfig.model_validate(toml_data)
            world_def = toml_config.worlds.get(self.world_name)

            if not world_def:
                raise ValueError(f"World '{self.world_name}' not found in {config_path}.")

            # This is a simplified version of the logic from the old `_validate_and_create_components`
            # It's scoped to just the world we need to build.
            # A more robust solution might involve a dedicated factory function.
            from dam.core import plugin_loader  # noqa: PLC0415
            from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: PLC0415

            all_plugins = plugin_loader.get_all_plugins(world_def.enabled_plugins)
            plugins_to_validate = world_def.enabled_plugins or all_plugins.keys()

            config_components: list[ConfigComponent] = []
            for plugin_name in plugins_to_validate:
                plugin = all_plugins.get(plugin_name)
                if not plugin:
                    continue

                settings_data = world_def.plugin_settings.get(plugin_name, {})
                settings_model_class = getattr(plugin, "Settings", None)
                component_class = getattr(plugin, "SettingsComponent", None)

                if not settings_model_class or not component_class:
                    continue

                # Perform validation and environment variable overrides
                env_prefix = f"DAM_WORLDS_{self.world_name.upper()}_PLUGIN_SETTINGS_{plugin_name.upper()}_"
                dynamic_settings = type(
                    f"{plugin_name}RuntimeSettings",
                    (settings_model_class, BaseSettings),
                    {"model_config": SettingsConfigDict(env_prefix=env_prefix, env_nested_delimiter="__")},
                )
                validated_settings = dynamic_settings(**settings_data)

                # Create the component instance
                component_data = validated_settings.model_dump()
                component_data["plugin_name"] = plugin_name
                config_components.append(component_class(**component_data))

            # Use the core factory to create the world
            new_world = create_world_from_components(self.world_name, config_components)
            # Add the app-specific plugin
            new_world.add_plugin(AppPlugin())

            return new_world

        except Exception:
            logger.exception("Failed to instantiate world '%s'", self.world_name)
            return None


global_state = GlobalState()


def get_world() -> World | None:
    """Get the current world from the global state."""
    return global_state.get_current_world()
