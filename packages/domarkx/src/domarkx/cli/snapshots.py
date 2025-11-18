"""The 'project' subcommand for the domarkx CLI."""

import uuid

import typer
from rich.console import Console

from domarkx.config import Settings
from domarkx.managers.session_manager import SessionManager
from domarkx.managers.workspace_manager import WorkspaceManager
from domarkx.plugins.docker_sandbox import DockerSandboxPlugin

project_app = typer.Typer()


def register(cli_app: typer.Typer, _: Settings) -> None:
    """Register the 'project' subcommand with the main CLI app."""
    cli_app.add_typer(project_app, name="project", help="Manage projects (sessions and workspaces).")


@project_app.command("import")
def import_project(
    # markdown_file: Annotated[pathlib.Path, typer.Argument(exists=True)],
) -> None:
    """Import a Markdown file to create a new session and workspace."""
    console = Console()

    # 1. Initialize Managers and Plugins
    session_manager = SessionManager()
    workspace_manager = WorkspaceManager()
    docker_plugin = DockerSandboxPlugin()
    # In a real app, these would be singletons or provided by a DI container.

    # 2. Parse the Markdown file
    # Note: The parser needs to be updated to extract resource configurations.
    # For now, we'll assume it returns a conversation and a list of resource configs.
    # parser = MarkdownParser()
    # conversation, resource_configs = parser.parse(markdown_file) # This is a placeholder

    # 3. Create Session and Workspace
    session_id = str(uuid.uuid4())
    workspace_id = str(uuid.uuid4())
    session = session_manager.create_session(session_id)
    # session.messages = conversation # This would be used with the real parser
    workspace = workspace_manager.create_workspace(workspace_id)

    console.print(f"Created Session: {session.session_id}")
    console.print(f"Created Workspace: {workspace.workspace_id}")

    # 4. Create Resources from parsed config
    # Placeholder for a docker resource config that would come from the Markdown file.
    docker_resource_config = {"image": "ubuntu:latest", "name": "my-sandbox"}
    # resource_configs = [docker_resource_config]

    # for config in resource_configs:
    #     if config.get("type") == "docker_sandbox": # This check would be more robust
    docker_resource = docker_plugin.create_resource(docker_resource_config)
    workspace.resources[docker_resource.resource_id] = docker_resource
    console.print(f"Created Docker Resource: {docker_resource.resource_id}")

    console.print("--- Project State ---")
    console.print("Session:")
    console.print(session.model_dump_json(indent=2))
    console.print("Workspace:")
    console.print(workspace.model_dump_json(indent=2))
