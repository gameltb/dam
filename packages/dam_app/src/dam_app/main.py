"""Main CLI entry point for the DAM application."""

import logging
import os
from pathlib import Path
from typing import Annotated

import tomli
import typer
from dam.core.config_loader import TOMLConfig, find_config_file

from dam_app.cli import assets, db, report, verify, worlds
from dam_app.logging_config import setup_logging
from dam_app.state import global_state

app = typer.Typer(
    name="dam-cli",
    help="Digital Asset Management System CLI",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

app.add_typer(assets.app, name="assets", help="Commands for managing assets.")
app.add_typer(verify.app, name="verify", help="Commands for verifying asset integrity.")
app.add_typer(db.app, name="db", help="Commands for database schema management.")
app.add_typer(report.app, name="report", help="Commands for generating reports.")
app.add_typer(worlds.app, name="world", help="Commands for dynamic world management.")


def list_worlds() -> list[str]:
    """List all worlds defined in the configuration file."""
    config_path = find_config_file()
    if not config_path:
        return []

    with config_path.open("rb") as f:
        toml_data = tomli.load(f)

    toml_config = TOMLConfig.model_validate(toml_data)
    if not toml_config.worlds:
        return []

    return list(toml_config.worlds.keys())


@app.command(name="list-worlds")
def cli_list_worlds():
    """List all worlds defined in the configuration file."""
    try:
        worlds = list_worlds()
        if not worlds:
            typer.secho("No worlds are defined in the configuration.", fg=typer.colors.YELLOW)
            return

        typer.echo("Defined worlds:")
        for world_name in worlds:
            is_active = world_name == global_state.world_name
            active_marker = " (active)" if is_active else ""
            typer.echo(f"  - {world_name}{active_marker}")

    except Exception as e:
        typer.secho(f"Error reading configuration: {e}", fg=typer.colors.RED)
        raise typer.Exit(1) from e


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    world: Annotated[
        str | None,
        typer.Option(
            "--world",
            "-w",
            help="Name of the ECS world to operate on.",
            envvar="DAM_CURRENT_WORLD",
        ),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to the configuration file (e.g., dam.toml).",
            envvar="DAM_CONFIG_FILE",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
):
    """Initialize the application, load configuration, and set the target world."""
    setup_logging(logging.INFO)

    # Store the config file path if provided, so it can be found by on-demand loaders.
    if config_file:
        os.environ["DAM_CONFIG_FILE"] = str(config_file)

    # Set the active world name. The actual World object will be instantiated
    # on-demand by get_world() when a command needs it.
    global_state.world_name = world

    # Validate that a world is specified for commands that require it.
    # `db` commands and `list-worlds` are exempt.
    requires_world = not (
        ctx.invoked_subcommand is None
        or (ctx.invoked_subcommand and "db" in ctx.invoked_subcommand)
        or ctx.invoked_subcommand == "list-worlds"
    )

    if requires_world:
        if not world:
            typer.secho("Error: A world must be specified with --world or DAM_CURRENT_WORLD.", fg=typer.colors.RED)
            raise typer.Exit(1)

        # The first access will trigger the lazy load and validation.
        active_world = global_state.get_current_world()
        if not active_world:
            typer.secho(
                f"Error: Failed to instantiate world '{world}'. Check configuration and logs.", fg=typer.colors.RED
            )
            raise typer.Exit(1)

    if ctx.invoked_subcommand is None:
        typer.echo("Welcome to the DAM CLI. Use --help to see available commands.")
        if global_state.world_name:
            typer.echo(f"Current world context: [bold cyan]{global_state.world_name}[/bold cyan]")
        else:
            typer.echo("No world context set. Use --world <name> for world-specific commands.")


def validate_world_and_config(ctx: typer.Context, world: str | None):
    """Validate that a world is specified and can be instantiated."""
    requires_world = not (
        ctx.invoked_subcommand is None
        or (ctx.invoked_subcommand and "db" in ctx.invoked_subcommand)
        or ctx.invoked_subcommand == "list-worlds"
    )

    if requires_world:
        if not world:
            raise typer.Exit(1)

        active_world = global_state.get_current_world()
        if not active_world:
            raise typer.Exit(1)


def run_cli_directly():
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    run_cli_directly()
