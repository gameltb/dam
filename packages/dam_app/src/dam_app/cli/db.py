"""Defines the CLI for database schema management using Alembic."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from dam.core.config import Config, get_dam_toml
from rich.console import Console

# --- Typer App Definition ---
app = typer.Typer(
    name="db",
    help="Commands for database schema management using Alembic.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()

# --- Helper Functions ---

# This points to the directory containing our alembic templates.
ALEMBIC_TEMPLATE_DIR = Path(__file__).parent.parent / "alembic_template"


def get_alembic_ini_path(world_name: str, config: Config) -> Path:
    """
    Get the path to the alembic.ini file for a given world.

    This function relies on the mandatory `paths.alembic` configuration.
    """
    world_def = config.worlds.get(world_name)
    if not world_def or "alembic" not in world_def.paths:
        console.print(
            f"[bold red]Error:[/] The `[worlds.{world_name}.paths]` table with an 'alembic' key is not defined in your dam.toml."
        )
        raise typer.Exit(1)

    return world_def.paths["alembic"].resolve() / "alembic.ini"


def _run_alembic_command(world_name: str, command: list[str]):
    """Set up the environment and run an Alembic command."""
    # First, load the config and find the alembic.ini path.
    try:
        config_path_str = os.getenv("DAM_CONFIG_FILE")
        config_path = Path(config_path_str) if config_path_str else None
        config = get_dam_toml().parse(config_path)
        if world_name not in config.worlds:
            console.print(f"[bold red]Error:[/] World '{world_name}' not found in configuration.")
            raise typer.Exit(1)

        alembic_ini_path = get_alembic_ini_path(world_name, config)
        if not alembic_ini_path.is_file():
            console.print(f"[bold red]Error:[/] alembic.ini not found at configured path: {alembic_ini_path}")
            console.print(f"Please run `dam-cli db init --world {world_name}` first.")
            raise typer.Exit(1)

    except FileNotFoundError as e:
        console.print("[bold red]Error:[/] Configuration file not found.")
        raise typer.Exit(1) from e

    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["DAM_WORLD_NAME"] = world_name
    # Ensure the project root is in PYTHONPATH so alembic's env.py can import `dam_app`
    project_root = Path(__file__).parent.parent.parent.parent.parent
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    full_command = [sys.executable, "-m", "alembic", "-c", str(alembic_ini_path), *command]
    console.print(f"[dim]Running command: {' '.join(full_command)}[/dim]")

    try:
        # Use subprocess.run without capturing output to stream directly to console
        result = subprocess.run(full_command, check=True, env=env, text=True)
        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(result.stderr)

    except subprocess.CalledProcessError as e:
        console.print("[bold red]Alembic command failed.[/bold red]")
        raise typer.Exit(1) from e
    except FileNotFoundError as e:
        console.print("[bold red]Error:[/] `alembic` command not found. Is it installed?")
        raise typer.Exit(1) from e


# --- CLI Commands ---


@app.command()
def init(
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to initialize migrations for.")],
):
    """Initialize the migration environment for a specific world."""
    console.print(f"Initializing migration environment for world: [bold cyan]{world}[/bold cyan]")
    try:
        config_path_str = os.getenv("DAM_CONFIG_FILE")
        config_path = Path(config_path_str) if config_path_str else None
        config = get_dam_toml().parse(config_path)
        if world not in config.worlds:
            console.print(f"[bold red]Error:[/] World '{world}' not found in configuration.")
            raise typer.Exit(1)

        # get_alembic_ini_path also validates the path is configured
        alembic_ini_path = get_alembic_ini_path(world, config)
        alembic_dir = alembic_ini_path.parent
        alembic_dir.mkdir(parents=True, exist_ok=True)

        # Copy the template files
        shutil.copyfile(ALEMBIC_TEMPLATE_DIR / "alembic.ini", alembic_dir / "alembic.ini")
        shutil.copyfile(ALEMBIC_TEMPLATE_DIR / "env.py", alembic_dir / "env.py")
        shutil.copyfile(ALEMBIC_TEMPLATE_DIR / "script.py.mako", alembic_dir / "script.py.mako")
        (alembic_dir / "versions").mkdir(exist_ok=True)

        console.print(f"[green]Successfully initialized.[/green] Migration files will be stored in: {alembic_dir}")

    except FileNotFoundError as e:
        console.print("[bold red]Error:[/] Configuration file not found.")
        raise typer.Exit(1) from e


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def revision(
    ctx: typer.Context,
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to create a revision for.")],
    message: Annotated[str | None, typer.Option("-m", "--message", help="Revision message.")] = None,
    autogenerate: Annotated[bool, typer.Option(help="Autogenerate revision from model changes.")] = False,
):
    """Create a new revision file."""
    command = ["revision"]
    if autogenerate:
        command.append("--autogenerate")
    if message:
        command.extend(["-m", message])
    command.extend(ctx.args)
    _run_alembic_command(world, command)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def upgrade(
    ctx: typer.Context,
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to upgrade.")],
    revision: Annotated[str, typer.Argument(help="Revision to upgrade to (e.g., 'head', '+1').")] = "head",
):
    """Upgrade to a later version."""
    command = ["upgrade", revision, *ctx.args]
    _run_alembic_command(world, command)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def downgrade(
    ctx: typer.Context,
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to downgrade.")],
    revision: Annotated[str, typer.Argument(help="Revision to downgrade to (e.g., '-1', 'base').")] = "-1",
):
    """Revert to a previous version."""
    command = ["downgrade", revision, *ctx.args]
    _run_alembic_command(world, command)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def history(
    ctx: typer.Context,
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to show history for.")],
):
    """List changeset scripts in chronological order."""
    command = ["history", *ctx.args]
    _run_alembic_command(world, command)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def current(
    ctx: typer.Context,
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to show current revision for.")],
):
    """Display the current revision for a database."""
    command = ["current", *ctx.args]
    _run_alembic_command(world, command)
