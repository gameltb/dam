"""Tools for managing chat sessions."""

import pathlib
import subprocess
from datetime import datetime
from typing import Any

from domarkx.config import settings
from domarkx.macro_expander import MacroExpander
from domarkx.tools.tool_factory import tool_handler


@tool_handler()
def create_session(template_name: str, session_name: str, parameters: dict[str, Any]) -> str:
    """
    Create a new session file from a template and parameters.

    Args:
        template_name (str): The name of the template (without .md extension).
        session_name (str): The name of the session to create (will be used as filename).
        parameters (dict[str, Any]): Dictionary of parameters to expand in the template. Must include any macros used in the template.

    Returns:
        str: Success message.

    Raises:
        FileNotFoundError: If the template file does not exist.

    Example:
        >>> create_session("default", "test_session", {"session_name": "Test Session", "user_prompt": "Hello"})
        "Session 'test_session' created from template 'default'."

    The session file will be created in the sessions directory and committed to git.

    """
    project_path = pathlib.Path(settings.project_path)
    template_path = project_path / "templates" / f"{template_name}.md"
    session_path = project_path / "sessions" / f"{session_name}.md"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    template_content = template_path.read_text(encoding="utf-8")

    expander = MacroExpander(base_dir=str(project_path / "templates"))

    override_parameters: dict[str, dict[str, Any]] = {}
    for key, value in parameters.items():
        override_parameters[key] = {"value": value}

    expanded_content = expander.expand(template_content, override_parameters=override_parameters)

    session_path.write_text(expanded_content, encoding="utf-8")

    # Add to git
    subprocess.run(["git", "add", str(session_path)], check=False, cwd=settings.project_path)
    subprocess.run(
        ["git", "commit", "-m", f"Create session {session_name}"],
        check=False,
        cwd=settings.project_path,
    )

    return f"Session '{session_name}' created from template '{template_name}'."


@tool_handler()
def send_message(session_name: str, message: str) -> str:
    """
    Append a message to a session file.

    Args:
        session_name (str): The name of the session file (without .md extension).
        message (str): The message to append.

    Returns:
        str: Success message.

    Raises:
        FileNotFoundError: If the session file does not exist.

    Example:
        >>> send_message("test_session", "This is a test message.")
        "Message sent to session 'test_session'."

    The message will be appended to the session file and committed to git.

    """
    session_path = pathlib.Path(settings.project_path) / "sessions" / f"{session_name}.md"

    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    with session_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n{message}")

    # Add to git
    subprocess.run(["git", "add", str(session_path)], check=False, cwd=settings.project_path)
    subprocess.run(
        ["git", "commit", "-m", f"Send message to session {session_name}"],
        check=False,
        cwd=settings.project_path,
    )

    return f"Message sent to session '{session_name}'."


@tool_handler()
def get_messages(session_name: str) -> str:
    """
    Get all messages/content from a session file.

    Args:
        session_name (str): The name of the session file (without .md extension).

    Returns:
        str: The full content of the session file.

    Raises:
        FileNotFoundError: If the session file does not exist.

    Example:
        >>> get_messages("test_session")
        "...session content..."

    """
    session_path = pathlib.Path(settings.project_path) / "sessions" / f"{session_name}.md"

    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    return session_path.read_text(encoding="utf-8")


@tool_handler()
def list_sessions() -> str:
    r"""
    List all available session files in the sessions directory.

    Returns:
        str: Newline-separated list of session filenames (ending with .md).

    Example:
        >>> list_sessions()
        "test_session.md\nanother_session.md"

    """
    sessions_path = pathlib.Path(settings.project_path) / "sessions"
    sessions = [p.name for p in sessions_path.iterdir() if p.is_file() and p.name.endswith(".md")]
    return "\n".join(sessions)


@tool_handler()
def archive_session(session_name: str, topic: str) -> str:
    """
    Archive a session file based on its topic and creation date.

    Args:
        session_name (str): The name of the session file (without .md extension).
        topic (str): The topic of the session.

    Returns:
        str: Success message.

    Raises:
        FileNotFoundError: If the session file does not exist.

    Example:
        >>> archive_session("test_session", "testing")
        "Session 'test_session' archived to 'archive/testing/2023-11-15_test_session.md'."

    The session file will be moved to the archive directory and the changes will be committed to git.

    """
    project_path = pathlib.Path(settings.project_path)
    session_path = project_path / "sessions" / f"{session_name}.md"
    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")

    # Get creation date
    creation_timestamp = session_path.stat().st_ctime
    creation_date = datetime.fromtimestamp(creation_timestamp).strftime("%Y-%m-%d")

    # Create archive directory
    archive_dir = project_path / "archive" / topic
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Move and rename session file
    new_session_name = f"{creation_date}_{session_name}.md"
    new_session_path = archive_dir / new_session_name
    session_path.rename(new_session_path)

    # Add to git
    subprocess.run(["git", "add", str(new_session_path)], check=False, cwd=settings.project_path)
    subprocess.run(["git", "rm", str(session_path)], check=False, cwd=settings.project_path)
    subprocess.run(
        ["git", "commit", "-m", f"Archive session {session_name} to {topic}"],
        check=False,
        cwd=settings.project_path,
    )

    return f"Session '{session_name}' archived to '{new_session_path}'."
