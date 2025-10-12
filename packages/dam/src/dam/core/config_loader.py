"""
Defines the Pydantic models and loading functions for the dam.toml config.

This module is responsible for:
1. Defining the basic structure of the `dam.toml` file.
2. Discovering all available plugins and their defined `SettingsModel`.
3. Loading the TOML file and validating the configuration for each plugin
   against its `SettingsModel`.
4. Supporting environment variable overrides for all plugin settings.
5. Creating instances of each plugin's `ConfigComponent` with the validated
   settings data.
"""

import importlib
from pathlib import Path
from typing import Any

import tomli
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dam.core import plugin_loader
from dam.models.config import ConfigComponent, SettingsModel

# --- Configuration Models ---


class WorldDefinition(BaseModel):
    """Defines the configuration for a single world instance in dam.toml."""

    plugin_settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plugin_settings", mode="before")
    @classmethod
    def ensure_core_plugin_settings_exist(cls, value: Any) -> Any:
        """Ensure that the 'core' plugin is configured for every world."""
        if not isinstance(value, dict) or "core" not in value:
            raise ValueError("Each world must have a `[worlds.<name>.plugin_settings.core]` table.")
        return value


class TOMLConfig(BaseModel):
    """Root Pydantic model for parsing the dam.toml file."""

    worlds: dict[str, WorldDefinition] = Field(default_factory=dict)

    @field_validator("worlds", mode="before")
    @classmethod
    def ensure_worlds_are_present(cls, value: Any) -> Any:
        """Ensure the configuration contains at least one world definition."""
        if not isinstance(value, dict) or not value:
            raise ValueError("Configuration must contain at least one `[worlds.<name>]` table.")
        return value


# --- Loading and Validation Logic ---


def _validate_and_create_components(
    toml_config: TOMLConfig,
) -> dict[str, dict[str, ConfigComponent]]:
    """
    Validates plugin settings and creates ConfigComponent instances.

    Iterates through the worlds and plugins defined in the TOML config,
    validates their settings using the appropriate SettingsModel, applies
    environment variable overrides, and creates the corresponding ConfigComponent.
    """
    all_plugins = plugin_loader.get_all_plugins()
    world_components: dict[str, dict[str, ConfigComponent]] = {}

    for world_name, world_def in toml_config.worlds.items():
        world_components[world_name] = {}
        for plugin_name, settings_data in world_def.plugin_settings.items():
            plugin = all_plugins.get(plugin_name)
            if not plugin:
                # This might happen if a plugin is configured in toml but not installed
                # or discoverable. The plugin_loader already logs a warning.
                continue

            settings_model_class = getattr(plugin, "Settings", None)
            if not settings_model_class or not issubclass(settings_model_class, SettingsModel):
                # Plugin exists but doesn't define a SettingsModel, so no config is expected.
                continue

            # Dynamically create a pydantic-settings class to handle env vars
            env_prefix = f"DAM_WORLDS_{world_name.upper()}_PLUGIN_SETTINGS_{plugin_name.upper()}_"
            DynamicSettings = type(
                f"{plugin_name}RuntimeSettings",
                (settings_model_class, BaseSettings),
                {
                    "model_config": SettingsConfigDict(
                        env_prefix=env_prefix,
                        env_nested_delimiter="__",
                        extra="ignore",
                    )
                },
            )

            # Instantiate with data from TOML, which will be overridden by env vars
            validated_settings = DynamicSettings(**settings_data)

            # Now, create and populate the corresponding ConfigComponent
            component_class = _get_component_class_for_plugin(plugin)
            if component_class:
                component_instance = component_class(**validated_settings.model_dump())
                world_components[world_name][plugin_name] = component_instance

    return world_components


def _get_component_class_for_plugin(plugin: "plugin_loader.Plugin") -> type[ConfigComponent] | None:
    """
    Finds the ConfigComponent subclass associated with a plugin by looking for
    a `SettingsComponent` attribute on the plugin class.
    """
    component_class = getattr(plugin, "SettingsComponent", None)
    if component_class and issubclass(component_class, ConfigComponent):
        return component_class
    return None


def load_and_validate_settings(
    config_path: Path | None = None,
) -> dict[str, dict[str, ConfigComponent]]:
    """
    Load, validate, and process the application configuration from a TOML file.

    This is the main entry point for configuration loading. It orchestrates
    reading the TOML file, validating its structure, and then dynamically
    validating each plugin's settings, resulting in a dictionary of
    ConfigComponent instances for each world.
    """
    found_path = config_path or _find_config_file()
    if not found_path:
        raise FileNotFoundError(
            "Configuration file 'dam.toml' or '.dam.toml' not found in current or parent directories."
        )

    with found_path.open("rb") as f:
        toml_data = tomli.load(f)

    # First, validate the basic structure of the TOML file
    toml_config = TOMLConfig.model_validate(toml_data)

    # Now, dynamically validate settings for all plugins and create components
    return _validate_and_create_components(toml_config)


def _find_config_file() -> Path | None:
    """Search for dam.toml or .dam.toml in the current dir and parents."""
    search_dir = Path.cwd()
    # Search up to the root directory
    while search_dir != search_dir.parent:
        for filename in ["dam.toml", ".dam.toml"]:
            p = search_dir / filename
            if p.is_file():
                return p
        search_dir = search_dir.parent
    return None
