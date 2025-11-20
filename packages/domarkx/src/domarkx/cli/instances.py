"""The 'instance' subcommand for the domarkx CLI."""

import uuid

import typer
from rich.console import Console

from domarkx.config import Settings
from domarkx.data.models import Message, TextBlock
from domarkx.managers.session_manager import SessionManager
from domarkx.managers.workspace_manager import WorkspaceManager
from domarkx.plugins.docker_sandbox import DockerSandboxPlugin

instance_app = typer.Typer()


def register(cli_app: typer.Typer, _: Settings) -> None:
    """Register the 'instance' subcommand with the main CLI app."""
    cli_app.add_typer(instance_app, name="instance", help="Manage session instances.")


@instance_app.command("run")
def run_instance() -> None:
    """Run a new session instance (demonstration)."""
    console = Console()

    # 1. Initialize Managers
    session_manager = SessionManager()
    workspace_manager = WorkspaceManager()
    docker_plugin = DockerSandboxPlugin()

    # 2. Create a new session and workspace
    session_id = str(uuid.uuid4())
    workspace_id = str(uuid.uuid4())
    session = session_manager.create_session(session_id)
    workspace = workspace_manager.create_workspace(workspace_id)
    console.print(f"Created Session: {session.session_id}")
    console.print(f"Created Workspace: {workspace.workspace_id}")

    # 3. Create a resource in the workspace
    docker_resource_config = {"image": "ubuntu:latest"}
    docker_resource = docker_plugin.create_resource(docker_resource_config)
    workspace.resources[docker_resource.resource_id] = docker_resource
    console.print(f"Created Docker Resource: {docker_resource.resource_id}")

    # 4. Simulate a simple conversation
    # User message
    user_message = Message(
        role="user",
        content=[TextBlock(value="List the files in the root directory.")],
        workspace_version_id=None,
    )
    session.messages.append(user_message)

    # Assistant executes a tool call
    tool_name = "run_command"
    tool_args = {"command": "ls -l /"}
    console.print(f"Executing tool '{tool_name}' with args: {tool_args}")
    result = docker_plugin.execute_tool(docker_resource.resource_id, tool_name, **tool_args)
    console.print(f"Tool execution result: {result}")

    # 5. Commit the new workspace version
    new_version_id = docker_plugin.commit_version(docker_resource.resource_id)
    console.print(f"Committed new workspace version: {new_version_id}")

    # 6. Create the assistant message and link it to the new version
    assistant_message = Message(
        role="assistant",
        content=[TextBlock(value=f"Execution finished with exit code {result['exit_code']}.")],
        workspace_version_id=new_version_id,
    )
    session.messages.append(assistant_message)
    session_manager.update_session(session)

    console.print("--- Final Session State ---")
    console.print(session.model_dump_json(indent=2))
