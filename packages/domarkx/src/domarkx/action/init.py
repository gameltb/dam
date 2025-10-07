"""The init command for domarkx."""
import logging
import pathlib
import shutil
import subprocess

import typer

from domarkx.config import Settings

logger = logging.getLogger(__name__)


def init_project(
    project_path_str: str = typer.Option(
        ".",
        "--path",
        "-p",
        help="The path to initialize the project in.",
    ),
    template_path_str: str | None = typer.Option(
        None,
        "--template",
        "-t",
        help="The path to a custom template to use for initialization.",
    ),
) -> None:
    """Initialize a new domarkx project."""
    project_path = pathlib.Path(project_path_str).resolve()
    if not project_path.exists():
        project_path.mkdir(parents=True)

    # Initialize Git repository
    subprocess.run(["git", "init"], check=False, cwd=project_path)

    # Add template handling
    if template_path_str:
        template_path = pathlib.Path(template_path_str)
        logger.info("Initializing project from template: %s", template_path)
        # TODO: Implement custom template copying
    else:
        logger.info("Initializing project with default template.")
        default_template_path = pathlib.Path(__file__).parent / ".." / "templates" / "default"
        for item_path in default_template_path.iterdir():
            dest_path = project_path / item_path.name
            if item_path.is_dir():
                shutil.copytree(item_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(item_path, dest_path)

    logger.info("Project initialized at: %s", project_path)


def register(app: typer.Typer, settings: Settings) -> None:
    """Register the init command with the Typer app."""

    @app.command(name="init")
    def init_command(  # type: ignore
        project_path: str = typer.Option(
            settings.DOMARKX_PROJECT_PATH,
            "--path",
            "-p",
            help="The path to initialize the project in.",
        ),
        template_path: str | None = typer.Option(
            None,
            "--template",
            "-t",
            help="The path to a custom template to use for initialization.",
        ),
    ) -> None:
        init_project(project_path, template_path)
