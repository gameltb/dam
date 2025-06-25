import json
import os
from typing import Dict, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorldConfig(BaseSettings):
    """Configuration for a single ECS world."""

    DATABASE_URL: str = Field("sqlite:///./default_dam.db")
    ASSET_STORAGE_PATH: str = Field("./default_dam_storage")

    model_config = SettingsConfigDict(extra="ignore")


class Settings(BaseSettings):
    """
    Application settings.
    Values are loaded from environment variables and/or a .env file.
    Individual worlds can be configured via a JSON string or a separate file.
    """

    # DAM_WORLDS can be a JSON string:
    # e.g., {"world1": {"DATABASE_URL": "...", "ASSET_STORAGE_PATH": "..."}, "world2": {...}}
    # or a path to a JSON file: e.g., /path/to/worlds_config.json
    DAM_WORLDS: str = Field(
        default='{"default": {"DATABASE_URL": "sqlite:///./dam.db", "ASSET_STORAGE_PATH": "./dam_storage"}}',
        validation_alias="DAM_WORLDS_CONFIG",
    )

    worlds: Dict[str, WorldConfig] = Field(default_factory=dict)

    DEFAULT_WORLD_NAME: Optional[str] = Field("default", validation_alias="DAM_DEFAULT_WORLD_NAME")

    TESTING_MODE: bool = Field(False, validation_alias="TESTING_MODE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields from .env
    )

    @model_validator(mode="before")
    @classmethod
    def load_worlds_config(cls, values):
        dam_worlds_str = values.get("DAM_WORLDS", values.get("dam_worlds"))  # pydantic v2 uses field name

        if not dam_worlds_str:
            # Fallback to default if DAM_WORLDS is empty or not set at all
            dam_worlds_str = (
                '{"default": {"DATABASE_URL": "sqlite:///./dam.db", "ASSET_STORAGE_PATH": "./dam_storage"}}'
            )
            # Also ensure DEFAULT_WORLD_NAME reflects this if it wasn't explicitly set
            if not values.get("DEFAULT_WORLD_NAME", values.get("default_world_name")):
                values["DEFAULT_WORLD_NAME"] = "default"  # or default_world_name for pydantic v2

        parsed_worlds: Dict[str, dict] = {}
        if dam_worlds_str:
            if os.path.exists(dam_worlds_str):
                try:
                    with open(dam_worlds_str, "r") as f:
                        parsed_worlds = json.load(f)
                except (IOError, json.JSONDecodeError) as e:
                    raise ValueError(f"Error reading or parsing worlds config file {dam_worlds_str}: {e}")
            else:
                try:
                    parsed_worlds = json.loads(dam_worlds_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error parsing DAM_WORLDS JSON string: {e}")

        if not isinstance(parsed_worlds, dict):
            raise ValueError("Worlds configuration must be a JSON object (dictionary).")

        # Ensure there's at least one world configuration, using defaults if necessary
        if not parsed_worlds:
            parsed_worlds = {"default": {"DATABASE_URL": "sqlite:///./dam.db", "ASSET_STORAGE_PATH": "./dam_storage"}}
            if "DEFAULT_WORLD_NAME" not in values and "default_world_name" not in values:  # pydantic v2
                values["DEFAULT_WORLD_NAME"] = "default"

        final_worlds = {}
        for name, config_dict in parsed_worlds.items():
            final_worlds[name] = WorldConfig(**config_dict)

        values["worlds"] = final_worlds

        # Determine default world name
        default_world_name = values.get("DEFAULT_WORLD_NAME", values.get("default_world_name"))
        if not default_world_name and "default" in final_worlds:
            values["DEFAULT_WORLD_NAME"] = "default"  # or default_world_name
        elif default_world_name and default_world_name not in final_worlds:
            raise ValueError(f"DEFAULT_WORLD_NAME '{default_world_name}' not found in configured worlds.")
        elif not default_world_name and final_worlds:
            # If no default is set and 'default' is not a key, pick the first one.
            first_world_name = next(iter(final_worlds))
            values["DEFAULT_WORLD_NAME"] = first_world_name  # or default_world_name
        elif not final_worlds:
            # This case should ideally be prevented by the logic ensuring at least one world.
            # If it somehow occurs, there's no valid default.
            values["DEFAULT_WORLD_NAME"] = None  # or default_world_name

        return values

    def get_world_config(self, world_name: Optional[str] = None) -> WorldConfig:
        """
        Retrieves the configuration for a specific world.
        If world_name is None, returns the default world configuration.
        Raises ValueError if the world is not found.
        """
        effective_world_name: str
        if world_name:
            effective_world_name = world_name
        elif self.DEFAULT_WORLD_NAME:
            effective_world_name = self.DEFAULT_WORLD_NAME
        elif self.worlds:
            effective_world_name = next(iter(self.worlds))  # Fallback to the first world if no default
        else:
            raise ValueError(
                "Cannot determine world: No specific world name provided, no default world name set, and no worlds configured."
            )

        if effective_world_name not in self.worlds:
            raise ValueError(
                f"World '{effective_world_name}' not found in configuration. "
                f"Attempted to find: '{effective_world_name}'. "
                f"Provided world_name argument to get_world_config(): '{world_name}'. "
                f"Current DEFAULT_WORLD_NAME: '{self.DEFAULT_WORLD_NAME}'. "
                f"Available worlds: {list(self.worlds.keys())}"
            )
        return self.worlds[effective_world_name]


# Create a single settings instance to be used throughout the application
settings = Settings()

# Example .env entries:
# DAM_WORLDS_CONFIG='{"main_db": {"DATABASE_URL": "postgresql://user:pass@host:port/main", "ASSET_STORAGE_PATH": "/mnt/assets/main"}, "archive_db": {"DATABASE_URL": "sqlite:///./archive.db", "ASSET_STORAGE_PATH": "./archive_storage"}}'
# DAM_DEFAULT_WORLD_NAME="main_db"
#
# Or using a file:
# DAM_WORLDS_CONFIG="/path/to/my_worlds_config.json"
#
# Content of /path/to/my_worlds_config.json:
# {
#   "main_db": {
#     "DATABASE_URL": "postgresql://user:pass@host:port/main",
#     "ASSET_STORAGE_PATH": "/mnt/assets/main"
#   },
#   "archive_db": {
#     "DATABASE_URL": "sqlite:///./archive.db",
#     "ASSET_STORAGE_PATH": "./archive_storage"
#   }
# }
#
# If DAM_WORLDS_CONFIG is not set, it will default to a single "default" world:
# {"default": {"DATABASE_URL": "sqlite:///./dam.db", "ASSET_STORAGE_PATH": "./dam_storage"}}
# and DAM_DEFAULT_WORLD_NAME will be "default".
