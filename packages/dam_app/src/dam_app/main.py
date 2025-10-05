import logging
import traceback
from typing import Optional

import typer
from dam.core import config as app_config
from dam.core.logging_config import setup_logging
from dam.core.transaction import WorldTransaction
from dam.core.transaction_manager import TransactionManager
from dam.core.world import (
    clear_world_registry,
    create_and_register_all_worlds_from_settings,
    get_all_registered_worlds,
)
from typing_extensions import Annotated

from dam_app.cli import assets, verify
from dam_app.state import get_world, global_state
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper(
    name="dam-cli",
    help="Digital Asset Management System CLI",
    add_completion=True,
    rich_markup_mode=None,
)

app.add_typer(assets.app, name="assets", help="Commands for managing assets.")
app.add_typer(verify.app, name="verify", help="Commands for verifying asset integrity.")


@app.command(name="list-worlds")
def cli_list_worlds():
    """Lists all configured and registered ECS worlds."""
    try:
        registered_worlds = get_all_registered_worlds()
        if not registered_worlds:
            typer.secho("No ECS worlds are currently registered or configured correctly.", fg=typer.colors.YELLOW)
            return
        typer.echo("Available ECS worlds:")
        for world_instance in registered_worlds:
            is_default = app_config.settings.DEFAULT_WORLD_NAME == world_instance.name
            default_marker = " (default)" if is_default else ""
            typer.echo(f"  - {world_instance.name}{default_marker}")
    except Exception as e:
        typer.secho(f"Error listing worlds: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        return


@app.command(name="setup-db")
async def setup_db(ctx: typer.Context):
    target_world = get_world()
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(f"Setting up database for world: '{target_world.name}'...")
    try:
        transaction_manager = target_world.get_context(WorldTransaction)
        assert isinstance(transaction_manager, TransactionManager)
        await transaction_manager.create_db_and_tables()
        typer.secho("Database setup complete.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during database setup: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    world: Annotated[
        Optional[str],
        typer.Option(
            "--world",
            "-w",
            help="Name of the ECS world to operate on.",
            envvar="DAM_CURRENT_WORLD",
        ),
    ] = None,
):
    setup_logging()
    try:
        initialized_worlds = create_and_register_all_worlds_from_settings(app_settings=app_config.settings)
    except Exception as e:
        typer.secho(f"Critical error: Could not initialize worlds: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    # Dynamically load all plugins
    plugins = {
        "AppPlugin": "dam_app.plugin",
        "FsPlugin": "dam_fs",
        "SourcePlugin": "dam_source",
        "SirePlugin": "dam_sire",
        "ImagePlugin": "dam_media_image",
        "AudioPlugin": "dam_media_audio",
        "TranscodePlugin": "dam_media_transcode",
        "ArchivePlugin": "dam_archive",
        "PspPlugin": "dam_psp",
        "SemanticPlugin": "dam_semantic",
    }
    for plugin_class_name, module_name in plugins.items():
        try:
            module = __import__(module_name, fromlist=[plugin_class_name])
            plugin_class = getattr(module, plugin_class_name)
            for world_instance in initialized_worlds:
                world_instance.add_plugin(plugin_class())
        except ImportError:
            logging.info(f"{module_name} plugin not installed. Skipping.")

    # Determine and set the target world
    target_world_name = world or app_config.settings.DEFAULT_WORLD_NAME
    global_state.world_name = target_world_name

    if not target_world_name:
        if ctx.invoked_subcommand and ctx.invoked_subcommand not in ["list-worlds"]:
            typer.secho("Error: No world specified and no default is set.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    elif not get_world():
        if ctx.invoked_subcommand and ctx.invoked_subcommand not in ["list-worlds"]:
            typer.secho(f"Error: World '{target_world_name}' not found or not configured.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    if ctx.invoked_subcommand is None:
        current_world = get_world()
        world_name = current_world.name if current_world else "None"
        typer.echo(f"Current world: '{world_name}'")


def run_cli_directly():
    clear_world_registry()
    app()


if __name__ == "__main__":
    run_cli_directly()
