import importlib
import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dam.core.config_loader import load_and_validate_settings
from dam.models import Base
from dam.plugins.core import CoreSettingsComponent
from sqlalchemy import engine_from_config, pool

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Get the world name from an environment variable, which will be set by our CLI.
WORLD_NAME = os.getenv("DAM_WORLD_NAME")
if not WORLD_NAME:
    raise ValueError("DAM_WORLD_NAME environment variable must be set to run migrations.")

# Load the main application configuration to find the world's settings.
config_path_str = os.getenv("DAM_CONFIG_FILE")
config_path = Path(config_path_str) if config_path_str else None
world_configs = load_and_validate_settings(config_path)

world_config = world_configs.get(WORLD_NAME)
if not world_config:
    raise ValueError(f"World '{WORLD_NAME}' not found in configuration.")

# Unpack the tuple of (plugins, components)
loaded_plugins, world_components = world_config

core_settings = next(
    (comp for comp in world_components.values() if isinstance(comp, CoreSettingsComponent)), None
)
if not core_settings:
    raise ValueError(f"CoreSettingsComponent not found for world '{WORLD_NAME}'.")

# Set the database URL for the current world. Alembic will use this.
config.set_main_option("sqlalchemy.url", core_settings.database_url)


def import_plugin_models() -> None:
    """
    Dynamically import the 'models' module from the plugins enabled for the current world.
    This populates Base.metadata with the tables required for this specific world instance.
    """
    plugin_names_to_load = set(loaded_plugins.keys())
    print(f"Loading models for world '{WORLD_NAME}' with plugins: {', '.join(sorted(plugin_names_to_load))}")

    for plugin_name in plugin_names_to_load:
        module_name = plugin_name.replace("-", "_")
        try:
            importlib.import_module(f"{module_name}.models")
            print(f"  - Successfully imported models from '{plugin_name}'")
        except ImportError:
            print(f"  - Info: No models module found for plugin '{plugin_name}'. Skipping.")


# Dynamically load models from enabled plugins.
import_plugin_models()

# The target_metadata is now populated with all tables for this world.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
