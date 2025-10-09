import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from dam_app.config import load_config

# --- Typer App Definition ---
app = typer.Typer(
    name="db",
    help="Commands for database schema management using Alembic.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()

# --- Helper Functions ---

DAM_HOME = Path.home() / ".dam"
WORLDS_DIR = DAM_HOME / "worlds"
# This points to the directory containing our alembic templates (env.py, etc.)
ALEMBIC_TEMPLATE_DIR = Path(__file__).parent.parent / "alembic"


def get_world_dir(world_name: str) -> Path:
    """Returns the root directory for a given world's migrations."""
    return WORLDS_DIR / world_name


def setup_alembic_env(world_name: str, world_dir: Path) -> Path:
    """
    Sets up the necessary directory structure and config file for running Alembic.
    This function is idempotent.

    Args:
        world_name: The name of the world.
        world_dir: The root directory for the world's migration environment.

    Returns:
        The path to the generated alembic.ini file.
    """
    versions_dir = world_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    # Copy the master templates into the world's directory if they don't exist.
    # This makes each world's migration environment self-contained.
    shutil.copyfile(ALEMBIC_TEMPLATE_DIR / "env.py", world_dir / "env.py")
    shutil.copyfile(ALEMBIC_TEMPLATE_DIR / "script.py.mako", world_dir / "script.py.mako")

    # Generate the alembic.ini file for this specific world
    alembic_ini_path = world_dir / "alembic.ini"
    ini_content = f"""
[alembic]
# The script location is the world's migration directory.
# Alembic will look for env.py and the 'versions' directory here.
script_location = {world_dir.absolute()}

# Template for new migration files
file_template = %%(rev)s_%%(slug)s

# Set to true if you don't want to generate .pyc files
sourceless = true

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = INFO
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
    alembic_ini_path.write_text(ini_content)
    return alembic_ini_path


def _run_alembic_command(world_name: str, command: list[str]):
    """Helper to set up the environment and run an Alembic command."""
    # First, ensure the world is valid by trying to load the config
    try:
        config_path_str = os.getenv("DAM_CONFIG_FILE")
        config_path = Path(config_path_str) if config_path_str else None
        config = load_config(config_path)
        if world_name not in config.worlds:
            console.print(f"[bold red]Error:[/] World '{world_name}' not found in configuration.")
            raise typer.Exit(1)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Configuration file not found.")
        raise typer.Exit(1)

    world_dir = get_world_dir(world_name)
    alembic_ini_path = setup_alembic_env(world_name, world_dir)

    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["DAM_WORLD_NAME"] = world_name
    # Ensure the project root is in PYTHONPATH so alembic's env.py can import `dam_app`
    project_root = Path(__file__).parent.parent.parent.parent.parent
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    full_command = [sys.executable, "-m", "alembic", "-c", str(alembic_ini_path)] + command
    console.print(f"[dim]Running command: {' '.join(full_command)}[/dim]")

    try:
        subprocess.run(full_command, check=True, env=env, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        console.print("[bold red]Alembic command failed.[/bold red]")
        console.print("[bold]Alembic stdout:[/bold]")
        console.print(e.stdout)
        console.print("[bold]Alembic stderr:[/bold]")
        console.print(e.stderr)
        raise typer.Exit(1)
    except FileNotFoundError:
        console.print("[bold red]Error:[/] `alembic` command not found. Is it installed?")
        raise typer.Exit(1)


# --- CLI Commands ---

@app.command()
def init(
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to initialize migrations for.")]
):
    """Initializes the migration environment for a specific world."""
    console.print(f"Initializing migration environment for world: [bold cyan]{world}[/bold cyan]")
    world_dir = get_world_dir(world)
    setup_alembic_env(world, world_dir)
    console.print(f"[green]Successfully initialized.[/green] Migration files are in: {world_dir}")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def revision(
    ctx: typer.Context,
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to create a revision for.")],
    message: Annotated[Optional[str], typer.Option("-m", "--message", help="Revision message.")] = None,
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
    command = ["upgrade", revision] + ctx.args
    _run_alembic_command(world, command)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def downgrade(
    ctx: typer.Context,
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to downgrade.")],
    revision: Annotated[str, typer.Argument(help="Revision to downgrade to (e.g., '-1', 'base').")] = "-1",
):
    """Revert to a previous version."""
    command = ["downgrade", revision] + ctx.args
    _run_alembic_command(world, command)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def history(
    ctx: typer.Context,
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to show history for.")],
):
    """List changeset scripts in chronological order."""
    command = ["history"] + ctx.args
    _run_alembic_command(world, command)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def current(
    ctx: typer.Context,
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to show current revision for.")],
):
    """Display the current revision for a database."""
    command = ["current"] + ctx.args
    _run_alembic_command(world, command)