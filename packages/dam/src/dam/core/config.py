"""Defines the Pydantic models and loading functions for the dam.toml config."""

from pathlib import Path
from typing import Any, cast

import tomli
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Plugin Settings Models ---


class PluginSettings(BaseSettings):
    """
    Base class for plugin-specific settings.

    Plugins should subclass this to define their own settings, which will be
    loaded from the `[worlds.<world_name>.plugin_settings.<plugin_name>]`
    section of the `dam.toml` file.
    """

    model_config = SettingsConfigDict(extra="forbid")


# --- Configuration Models ---
# These are Pydantic models for parsing and validating the .toml file.
# They are not database components themselves.


class PluginConfig(BaseModel):
    """Define the plugin configuration for a world."""

    names: list[str] = Field(default_factory=list)


class DatabaseConfig(BaseModel):
    """Define the database configuration for a world."""

    url: str


class WorldDefinition(BaseModel):
    """Define the complete configuration for a single world instance."""

    db: DatabaseConfig
    plugins: PluginConfig
    paths: dict[str, Path] = Field(default_factory=dict)
    plugin_settings: dict[str, Any] = Field(default_factory=dict)


# --- Top-level Configuration Model ---


class DamToolConfig(BaseModel):
    """
    Model for the `[tool.dam]` section in `pyproject.toml`.

    This can be used for project-wide settings that are not world-specific.
    """

    TRANSCODING_TEMP_DIR: Path = Path("temp/dam_transcodes")


class Config(BaseModel):
    """Root model for the application's configuration file (`dam.toml`)."""

    worlds: dict[str, WorldDefinition] = Field(default_factory=dict)
    """A mapping from world names to their configurations."""

    @field_validator("worlds", mode="before")
    @classmethod
    def ensure_worlds_are_present(cls, value: Any) -> dict[str, Any]:
        """Ensure the configuration contains at least one world definition."""
        if not isinstance(value, dict) or not value:
            raise ValueError("Configuration must contain at least one `[worlds.<name>]` table.")
        return cast(dict[str, Any], value)


# --- Loading Function ---


class DamToml:
    """A helper class to find and parse the `dam.toml` file."""

    def __init__(self, start_dir: Path | None = None, filename: str = "dam.toml"):
        """Initialize the DamToml helper."""
        self.start_dir = start_dir or Path.cwd()
        self.filename = filename
        self._config: Config | None = None
        self._config_path: Path | None = None
        self._tool_config: DamToolConfig | None = None

    def find(self) -> Path | None:
        """
        Find the configuration file by searching up from the start directory.

        Returns:
            The path to the found configuration file, or None if not found.

        """
        if self._config_path:
            return self._config_path

        search_dir = self.start_dir
        while search_dir != search_dir.parent:
            for fname in [self.filename, f".{self.filename}"]:
                p = search_dir / fname
                if p.is_file():
                    self._config_path = p
                    return p
            search_dir = search_dir.parent
        return None

    def parse(self, config_path: Path | None = None) -> Config:
        """
        Load, validate, and cache the configuration from a TOML file.

        Args:
            config_path: An optional, explicit path to the configuration file.
                If not provided, the file will be searched for.

        Returns:
            The loaded and validated configuration object.

        Raises:
            FileNotFoundError: If no configuration file can be found.

        """
        if self._config and not config_path:
            return self._config

        path_to_load = config_path or self.find()
        if not path_to_load or not path_to_load.is_file():
            raise FileNotFoundError(f"Configuration file '{self.filename}' not found.")

        with path_to_load.open("rb") as f:
            data = tomli.load(f)

        self._config = Config.model_validate(data)
        self._config_path = path_to_load
        return self._config

    def get_tool_config(self) -> DamToolConfig:
        """
        Find and parse the `[tool.dam]` section from `pyproject.toml`.

        Returns:
            The loaded and validated tool configuration object.

        """
        if self._tool_config:
            return self._tool_config

        pyproject_path = self.find_pyproject_toml()
        if not pyproject_path:
            # Return a default config if pyproject.toml is not found
            self._tool_config = DamToolConfig()
            return self._tool_config

        with pyproject_path.open("rb") as f:
            pyproject_data = tomli.load(f)

        tool_dam_data = pyproject_data.get("tool", {}).get("dam", {})
        self._tool_config = DamToolConfig.model_validate(tool_dam_data)
        return self._tool_config

    def find_pyproject_toml(self) -> Path | None:
        """Find the `pyproject.toml` file."""
        search_dir = self.start_dir
        while search_dir != search_dir.parent:
            p = search_dir / "pyproject.toml"
            if p.is_file():
                return p
            search_dir = search_dir.parent
        return None


_dam_toml_cache = DamToml()


def get_dam_toml() -> DamToml:
    """Get the singleton instance of the DamToml helper."""
    return _dam_toml_cache
