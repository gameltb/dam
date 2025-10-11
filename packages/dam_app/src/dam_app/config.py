"""
Provides a consistent way to load the dam.toml configuration.

This is for the dam-app, particularly for tools like Alembic that are run from the
command line and need to locate the project's configuration file.
"""

from pathlib import Path

from dam.core.config import Config, get_dam_toml


def load_config(config_path: Path | None = None) -> Config:
    """
    Load, validate, and cache the configuration from a dam.toml file.

    This function is a simple wrapper around the core DAM configuration loading
    logic, making it easier to access from different parts of the dam-app.

    Args:
        config_path: An optional, explicit path to the configuration file.
            If not provided, the file will be searched for starting from the
            current working directory.

    Returns:
        The loaded and validated configuration object.

    """
    dam_toml = get_dam_toml()
    return dam_toml.parse(config_path)
