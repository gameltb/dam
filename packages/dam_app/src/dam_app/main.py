# --- Framework Imports for Systems ---
import logging
import traceback  # Import traceback for detailed error logging
from typing import List, Optional

import typer  # Ensure typer is imported for annotations like typer.Context
from dam.core import config as app_config
from dam.core.logging_config import setup_logging
from dam.core.world import (
    World,
    clear_world_registry,
    create_and_register_all_worlds_from_settings,
    get_all_registered_worlds,
)
from dam.functions import ecs_functions as dam_ecs_functions
from typing_extensions import Annotated

from dam_app.cli import assets, systems
from dam_app.state import get_world, global_state
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper(
    name="dam-cli",
    help="Digital Asset Management System CLI",
    add_completion=True,
    rich_markup_mode=None,
)


app.add_typer(assets.app, name="assets", help="Commands for managing assets.")
app.add_typer(systems.app, name="systems", help="Commands for running systems.")


@app.command(name="list-worlds")
def cli_list_worlds():
    """Lists all configured and registered ECS worlds."""
    try:
        registered_worlds = get_all_registered_worlds()
        if not registered_worlds:
            typer.secho("No ECS worlds are currently registered or configured correctly.", fg=typer.colors.YELLOW)
            typer.echo(
                "Check your DAM_WORLDS_CONFIG environment variable (JSON string or file path) and application logs for errors."
            )
            return

        typer.echo("Available ECS worlds:")
        for world_instance in registered_worlds:
            is_default = app_config.settings.DEFAULT_WORLD_NAME == world_instance.name
            default_marker = " (default)" if is_default else ""
            typer.echo(f"  - {world_instance.name}{default_marker}")

        configured_default = app_config.settings.DEFAULT_WORLD_NAME
        if configured_default:
            if not any(w.name == configured_default for w in registered_worlds):
                typer.secho(
                    f"\nWarning: The configured default world '{configured_default}' was not successfully registered. It might have configuration issues.",
                    fg=typer.colors.YELLOW,
                )
        elif registered_worlds:
            typer.secho(
                "\nNote: No specific default world is set. Operations might require explicit --world.",
                fg=typer.colors.YELLOW,
            )

    except Exception as e:
        typer.secho(f"Error listing worlds: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        return


@app.command(name="setup-db")
async def setup_db(ctx: typer.Context):
    target_world = get_world()
    if not target_world:
        typer.secho(
            f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    typer.echo(f"Setting up database for world: '{target_world.name}'...")
    try:
        await target_world.create_db_and_tables()
        typer.secho(f"Database setup complete for world '{target_world.name}'.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during database setup for world '{target_world.name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="show-entity")
async def cli_show_entity(
    ctx: typer.Context,
    entity_id: Annotated[
        int,
        typer.Argument(
            ...,
            help="The ID of the entity to show.",
        ),
    ],
):
    """
    Shows all components of a given entity.
    """
    target_world = get_world()
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    async with target_world.db_session_maker() as session:
        try:
            components = await dam_ecs_functions.get_all_components_for_entity(session, entity_id)
            if not components:
                typer.secho(f"No components found for entity {entity_id}", fg=typer.colors.YELLOW)
                return

            typer.secho(f"Components for entity {entity_id}:", fg=typer.colors.GREEN)
            for component in components:
                typer.echo(f"  - {component.__class__.__name__}:")
                for key, value in component.__dict__.items():
                    if not key.startswith("_"):
                        typer.echo(f"    - {key}: {value}")

        except Exception as e:
            typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            raise typer.Exit(code=1)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    world: Annotated[
        Optional[str],
        typer.Option(
            "--world",
            "-w",
            help="Name of the ECS world to operate on. Uses default world if not specified.",
            envvar="DAM_CURRENT_WORLD",
        ),
    ] = None,
):
    setup_logging()
    initialized_worlds: List[World] = []
    try:
        initialized_worlds = create_and_register_all_worlds_from_settings(app_settings=app_config.settings)
    except Exception as e:
        typer.secho(f"Critical error: Could not initialize worlds from settings: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    from dam_app.plugin import AppPlugin

    for world_instance in initialized_worlds:
        world_instance.add_plugin(AppPlugin())

    try:
        from dam_fs import FsPlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(FsPlugin())
    except ImportError:
        logging.info("dam_fs plugin not installed. Skipping fs functionality.")

    try:
        from dam_source import SourcePlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(SourcePlugin())
    except ImportError:
        logging.info("dam_source plugin not installed. Skipping source functionality.")

    try:
        from dam_sire import SirePlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(SirePlugin())
    except ImportError:
        logging.info("dam_sire plugin not installed. Skipping sire functionality.")

    try:
        from dam_media_image import ImagePlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(ImagePlugin())
    except ImportError:
        logging.info("dam_media_image plugin not installed. Skipping image functionality.")

    try:
        from dam_media_audio import AudioPlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(AudioPlugin())
    except ImportError:
        logging.info("dam_media_audio plugin not installed. Skipping audio functionality.")

    try:
        from dam_media_transcode import TranscodePlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(TranscodePlugin())
    except ImportError:
        logging.info("dam_media_transcode plugin not installed. Skipping transcode functionality.")

    try:
        from dam_archive import ArchivePlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(ArchivePlugin())
    except ImportError:
        logging.info("dam_archive plugin not installed. Skipping archive functionality.")

    try:
        from dam_psp import PspPlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(PspPlugin())
    except ImportError:
        logging.info("dam_psp plugin not installed. Skipping PSP ISO ingestion functionality.")

    try:
        from dam_semantic import SemanticPlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(SemanticPlugin())

    except ImportError:
        logging.info("dam_semantic plugin not installed. Skipping semantic search functionality.")

    target_world_name_candidate: Optional[str] = None
    if world:
        target_world_name_candidate = world
    elif app_config.settings.DEFAULT_WORLD_NAME:
        target_world_name_candidate = app_config.settings.DEFAULT_WORLD_NAME

    global_state.world_name = target_world_name_candidate

    if global_state.world_name:
        selected_world_instance = get_world()
        if selected_world_instance:
            if ctx.invoked_subcommand:
                typer.echo(f"Operating on world: '{global_state.world_name}'")
        else:
            if ctx.invoked_subcommand not in ["list-worlds"]:
                typer.secho(
                    f"Error: World '{global_state.world_name}' is specified or default, but it's not correctly configured or registered.",
                    fg=typer.colors.RED,
                )
                registered_worlds_list = get_all_registered_worlds()
                if registered_worlds_list:
                    typer.echo("Available correctly configured worlds:")
                    for w_instance in registered_worlds_list:
                        typer.echo(f"  - {w_instance.name}")
                else:
                    typer.echo("No worlds appear to be correctly configured.")
                raise typer.Exit(code=1)
    elif ctx.invoked_subcommand not in ["list-worlds"]:
        typer.secho(
            "Error: No ECS world specified and no default world could be determined. "
            "Use --world <world_name>, set DAM_DEFAULT_WORLD_NAME, or configure a 'default' world in DAM_WORLDS_CONFIG.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if ctx.invoked_subcommand is None:
        current_selection_info = (
            f"Current default/selected world: '{global_state.world_name}' (Use --world to change)"
            if global_state.world_name
            else "No default world selected. Use --world <world_name> or see 'list-worlds'."
        )
        typer.echo(current_selection_info)
        if not get_all_registered_worlds() and not app_config.settings.worlds:
            typer.secho("No DAM worlds seem to be configured. Please set DAM_WORLDS_CONFIG.", fg=typer.colors.YELLOW)


def run_cli_directly():
    clear_world_registry()
    app()


if __name__ == "__main__":
    run_cli_directly()
