import logging
import os
import shutil
import subprocess

import typer

from domarkx.config import Settings


def init_project(
    project_path: str = typer.Option(
        ".",
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
    """Initializes a new domarkx project."""
    project_path = os.path.abspath(project_path)
    if not os.path.exists(project_path):
        os.makedirs(project_path)

    # Initialize Git repository
    subprocess.run(["git", "init"], check=False, cwd=project_path)

    # Add template handling
    if template_path:
        logging.info(f"Initializing project from template: {template_path}")
        # TODO: Implement custom template copying
    else:
        logging.info("Initializing project with default template.")
        default_template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "default")
        for item in os.listdir(default_template_path):
            s = os.path.join(default_template_path, item)
            d = os.path.join(project_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, False, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    logging.info(f"Project initialized at: {project_path}")


def register(app: typer.Typer, settings: Settings) -> None:
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
