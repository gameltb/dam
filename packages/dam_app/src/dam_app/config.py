from pathlib import Path
from typing import Any

import tomli
from pydantic import BaseModel, Field, field_validator

# --- Configuration Models ---
# These are Pydantic models for parsing and validating the .toml file.
# They are not database components themselves.

class PluginConfig(BaseModel):
    """Defines the plugin configuration for a world."""
    names: list[str] = Field(default_factory=list)


class DatabaseConfig(BaseModel):
    """Defines the database configuration for a world."""
    url: str


class WorldDefinition(BaseModel):
    """
    Defines the complete configuration for a single world instance.
    """
    db: DatabaseConfig
    plugins: PluginConfig
    paths: dict[str, Path] = Field(default_factory=dict)


# --- Top-level Configuration Model ---

class Config(BaseModel):
    """Root model for the application's configuration file."""

    worlds: dict[str, WorldDefinition] = Field(default_factory=dict)
    """A mapping from world names to their configurations."""

    @field_validator("worlds", mode="before")
    @classmethod
    def ensure_worlds_are_present(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not value:
            raise ValueError("Configuration must contain at least one `[worlds.<name>]` table.")
        return value


# --- Loading Function ---

_cached_config: Config | None = None


def load_config(config_path: Path | None = None) -> Config:
    """
    Loads, validates, and caches the application configuration from a TOML file.

    The function searches for a `dam.toml` or `.dam.toml` file in the
    current directory and its parent directories. An explicit path can also be
    provided.

    Args:
        config_path: An optional, explicit path to a TOML configuration file.

    Returns:
        The loaded and validated configuration object.

    Raises:
        FileNotFoundError: If no configuration file can be found.
        tomli.TOMLDecodeError: If the file is not valid TOML.
        pydantic.ValidationError: If the configuration does not match the schema.
    """
    global _cached_config
    # If a specific path is given, always reload. Otherwise, use cache if available.
    if _cached_config and not config_path:
        return _cached_config

    found_path = config_path

    if not found_path:
        # Auto-discover config file
        search_dir = Path.cwd()
        while search_dir != search_dir.parent:
            for filename in ["dam.toml", ".dam.toml"]:
                p = search_dir / filename
                if p.is_file():
                    found_path = p
                    break
            if found_path:
                break
            search_dir = search_dir.parent

    if not found_path or not found_path.is_file():
        raise FileNotFoundError(f"Configuration file not found.")

    with open(found_path, "rb") as f:
        data = tomli.load(f)

    config = Config.model_validate(data)

    # Cache the config only if it was loaded via auto-discovery
    if not config_path:
        _cached_config = config

    return config