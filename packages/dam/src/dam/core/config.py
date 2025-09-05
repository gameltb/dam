import json
import logging
import os
from typing import Any, Dict, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class WorldConfig(BaseSettings):
    """Configuration for a single ECS world."""

    name: str  # Name of the world, will be set from the key in the parent Settings.worlds dict
    DATABASE_URL: str = Field("postgresql+psycopg://postgres:postgres@localhost:5432/default_dam")
    ASSET_STORAGE_PATH: str = Field("./default_dam_storage")
    # Add other world-specific configurations here as needed
    # e.g., specific API keys, function endpoints for this world

    model_config = SettingsConfigDict(extra="ignore")


_DEFAULT_WORLD_CONFIG_JSON = '{"default": {"DATABASE_URL": "postgresql+psycopg://postgres:postgres@localhost:5432/dam", "ASSET_STORAGE_PATH": "./dam_storage"}}'


class Settings(BaseSettings):
    """
    Application settings.
    Values are loaded from environment variables and/or a .env file.
    Manages configurations for multiple ECS worlds.
    """

    DAM_WORLDS_CONFIG: str = Field(
        default=_DEFAULT_WORLD_CONFIG_JSON,
        description="Path to a JSON file or a JSON string defining world configurations.",
    )

    worlds: Dict[str, WorldConfig] = Field(default_factory=dict, description="Dictionary of configured worlds.")

    DEFAULT_WORLD_NAME: Optional[str] = Field(
        "default",
        validation_alias="DAM_DEFAULT_WORLD_NAME",
        description="The name of the world to use by default if not specified.",
    )

    TESTING_MODE: bool = Field(False, validation_alias="TESTING_MODE", description="Enable testing mode behaviors.")

    TRANSCODING_TEMP_DIR: str = Field(  # Changed to str, Path will be constructed in code
        default="temp/dam_transcodes",
        validation_alias="DAM_TRANSCODING_TEMP_DIR",
        description="Temporary directory for transcoding operations before ingestion.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="before")
    @classmethod
    def _load_and_process_worlds_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Loads world configurations from the source specified by DAM_WORLDS_CONFIG.
        This source can be a file path to a JSON file or a direct JSON string.
        """
        config_source = values.get("DAM_WORLDS_CONFIG", values.get("dam_worlds_config"))

        if not config_source:
            logger.warning("DAM_WORLDS_CONFIG not set, using default world configuration.")
            config_source = _DEFAULT_WORLD_CONFIG_JSON
            # Ensure DEFAULT_WORLD_NAME reflects this if it wasn't explicitly set
            if not values.get("DEFAULT_WORLD_NAME", values.get("default_world_name")):
                values["DEFAULT_WORLD_NAME"] = "default"

        raw_world_configs: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(config_source):
            try:
                with open(config_source, "r") as f:
                    raw_world_configs = json.load(f)
                logger.info(f"Loaded world configurations from file: {config_source}")
            except (IOError, json.JSONDecodeError) as e:
                raise ValueError(f"Error reading or parsing worlds config file {config_source}: {e}")
        else:
            try:
                raw_world_configs = json.loads(config_source)
                logger.info("Loaded world configurations from JSON string.")
            except json.JSONDecodeError as e:
                # If it's not a valid path and not valid JSON, it might be a simple string meant for default.
                # This path is less likely if default is handled above.
                logger.warning(
                    f"DAM_WORLDS_CONFIG_SOURCE '{config_source}' is not a valid file path or JSON string. Attempting default. Error: {e}"
                )
                raw_world_configs = json.loads(_DEFAULT_WORLD_CONFIG_JSON)
                if not values.get("DEFAULT_WORLD_NAME", values.get("default_world_name")):
                    values["DEFAULT_WORLD_NAME"] = "default"

        if not isinstance(raw_world_configs, dict):
            raise ValueError("Worlds configuration must be a JSON object (dictionary of world_name: world_config).")

        if not raw_world_configs:  # Ensure there's at least one world
            logger.warning("No worlds found in configuration source. Adding a default world.")
            raw_world_configs = json.loads(_DEFAULT_WORLD_CONFIG_JSON)
            if not values.get("DEFAULT_WORLD_NAME", values.get("default_world_name")):
                values["DEFAULT_WORLD_NAME"] = "default"

        final_worlds_dict: Dict[str, WorldConfig] = {}
        for name, config_data in raw_world_configs.items():
            if not isinstance(config_data, dict):
                raise ValueError(f"Configuration for world '{name}' must be a dictionary.")
            # Add the world's name to its config data, so WorldConfig can store it.
            config_data_with_name = {"name": name, **config_data}
            final_worlds_dict[name] = WorldConfig(**config_data_with_name)

        values["worlds"] = final_worlds_dict

        # Validate or determine DEFAULT_WORLD_NAME
        # Explicitly check env var as validator 'values' might not fully reflect it yet for aliased fields
        env_default_world = os.environ.get("DAM_DEFAULT_WORLD_NAME")
        default_world_name_val = env_default_world or values.get("DEFAULT_WORLD_NAME", values.get("default_world_name"))

        if not final_worlds_dict:  # Should be caught by earlier check, but defensive
            logger.error("No worlds configured after processing. This is unexpected.")
            values["DEFAULT_WORLD_NAME"] = None
            return values

        if default_world_name_val:
            if default_world_name_val not in final_worlds_dict:
                raise ValueError(
                    f"DEFAULT_WORLD_NAME '{default_world_name_val}' is set but not found in the configured worlds: {list(final_worlds_dict.keys())}"
                )
            values["DEFAULT_WORLD_NAME"] = default_world_name_val  # Ensure values dict is updated
        elif "default" in final_worlds_dict:
            values["DEFAULT_WORLD_NAME"] = "default"
            logger.info("DEFAULT_WORLD_NAME not explicitly set, using 'default' world.")
        else:
            # No explicit default, 'default' world doesn't exist, pick the first one alphabetically.
            first_world_by_name = sorted(final_worlds_dict.keys())[0]
            values["DEFAULT_WORLD_NAME"] = first_world_by_name
            logger.info(
                f"DEFAULT_WORLD_NAME not set and 'default' world not found. Using first available world: '{first_world_by_name}'."
            )

        logger.debug(f"Final processed worlds: {list(values['worlds'].keys())}")
        logger.debug(f"Default world name set to: {values['DEFAULT_WORLD_NAME']}")
        return values

    def get_world_config(self, world_name: Optional[str] = None) -> WorldConfig:
        """
        Retrieves the configuration for a specific world.
        If world_name is None, returns the configuration for the default world.
        Raises ValueError if the specified or default world is not found.
        """
        target_world_name = world_name or self.DEFAULT_WORLD_NAME

        if not target_world_name:
            # This case should ideally be prevented by the validator ensuring a default world exists if any worlds are configured.
            if not self.worlds:
                raise ValueError("No worlds are configured, and no world name was specified.")
            raise ValueError("Cannot determine world: No world name provided and no default world name is set.")

        if target_world_name not in self.worlds:
            available_worlds = list(self.worlds.keys())
            error_message = (
                f"World '{target_world_name}' not found in configuration. "
                f"Requested: '{target_world_name}' (explicitly: '{world_name}', default: '{self.DEFAULT_WORLD_NAME}'). "
                f"Available worlds: {available_worlds}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        return self.worlds[target_world_name]

    def get_all_world_names(self) -> list[str]:
        """Returns a list of names of all configured worlds."""
        return list(self.worlds.keys())


# Global settings instance, initialized when this module is imported.
# Applications should import this instance.
settings = Settings()

# Example .env entries:
# DAM_WORLDS_CONFIG='{"main_db": {"DATABASE_URL": "postgresql://user:pass@host:port/main", "ASSET_STORAGE_PATH": "/mnt/assets/main"}, "archive_db": {"DATABASE_URL": "sqlite:///./archive.db", "ASSET_STORAGE_PATH": "./archive_storage"}}'
# DAM_DEFAULT_WORLD_NAME="main_db"
#
# Or using a file (recommended for complex setups):
# DAM_WORLDS_CONFIG="/path/to/my_worlds_config.json"
#
# Content of /path/to/my_worlds_config.json:
# {
#   "main_processing": {
#     "DATABASE_URL": "postgresql://user:pass@host:port/main_db",
#     "ASSET_STORAGE_PATH": "/srv/dam/assets_main"
#   },
#   "archive_cold_storage": {
#     "DATABASE_URL": "sqlite:///./archive.sqlite.db",
#     "ASSET_STORAGE_PATH": "/mnt/archive/dam_assets_cold"
#   },
#   "testing_world": {
#     "DATABASE_URL": "sqlite:///:memory:",
#     "ASSET_STORAGE_PATH": "/tmp/dam_test_assets"
#   }
# }
#
# If DAM_WORLDS_CONFIG is not set, it defaults to a single world named "default":
# {"default": {"DATABASE_URL": "sqlite:///./dam.db", "ASSET_STORAGE_PATH": "./dam_storage"}}
# and DAM_DEFAULT_WORLD_NAME will be "default".
