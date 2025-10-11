import asyncio
import importlib
import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from dam.core.config import get_dam_toml
from dam.models import Base

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
# It uses DAM_CONFIG_FILE env var if set, which our CLI does.
config_path_str = os.getenv("DAM_CONFIG_FILE")
config_path = Path(config_path_str) if config_path_str else None
app_config = get_dam_toml().parse(config_path)

world_definition = app_config.worlds.get(WORLD_NAME)

if not world_definition:
    raise ValueError(f"World '{WORLD_NAME}' not found in configuration.")


# Set the database URL for the current world. Alembic will use this.
config.set_main_option("sqlalchemy.url", world_definition.db.url)


def import_plugin_models() -> None:
    """
    Dynamically import the 'models' module from the plugins enabled for the current world.

    This populates Base.metadata with the tables required for this specific world instance.
    """
    # This assertion helps pyright understand that world_definition is not None here.
    assert world_definition is not None
    plugin_names_to_load = set(["dam"] + world_definition.plugins.names)

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
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=f"alembic_version_{WORLD_NAME}",
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run the migrations."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=f"alembic_version_{WORLD_NAME}",
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    url = config.get_main_option("sqlalchemy.url")
    assert url, "sqlalchemy.url must be set in alembic.ini or via config"
    connectable = create_async_engine(
        url,
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
