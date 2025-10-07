"""Tests for the session management tools."""

import pathlib
from typing import Any

from domarkx.config import settings
from domarkx.tools.session_management import (
    create_session,
    get_messages,
    list_sessions,
    send_message,
)


def test_create_session(tmp_path: Any) -> None:
    """Test creating a new session from a template."""
    project_path = pathlib.Path(tmp_path) / "test_project"
    (project_path / "sessions").mkdir(parents=True)
    (project_path / "templates").mkdir(parents=True)
    # Write the template file using the domarkx macro format
    (project_path / "templates" / "default.md").write_text(
        "# [@session_name](domarkx://set?value=default)\n\n[@user_prompt](domarkx://set?value=default)"
    )
    settings.DOMARKX_PROJECT_PATH = str(project_path)
    result = create_session("default", "test_session", {"session_name": "Test Session", "user_prompt": "Hello"})
    assert result == "Session 'test_session' created from template 'default'."
    session_file = project_path / "sessions" / "test_session.md"
    assert session_file.exists()
    content = session_file.read_text()
    assert "Test Session" in content
    assert "Hello" in content


def test_send_message(tmp_path: Any) -> None:
    """Test sending a message to a session."""
    project_path = pathlib.Path(tmp_path) / "test_project"
    settings.DOMARKX_PROJECT_PATH = str(project_path)
    # Ensure the session file exists
    (project_path / "sessions").mkdir(parents=True, exist_ok=True)
    session_file = project_path / "sessions" / "test_session.md"
    session_file.write_text("Test Session\nHello")
    result = send_message("test_session", "This is a test message.")
    assert result == "Message sent to session 'test_session'."
    content = session_file.read_text()
    assert "This is a test message." in content


def test_get_messages(tmp_path: Any) -> None:
    """Test retrieving messages from a session."""
    project_path = pathlib.Path(tmp_path) / "test_project"
    settings.DOMARKX_PROJECT_PATH = str(project_path)
    (project_path / "sessions").mkdir(parents=True, exist_ok=True)
    session_file = project_path / "sessions" / "test_session.md"
    session_file.write_text("Test Session\nHello\nThis is a test message.")
    content = get_messages("test_session")
    assert "Test Session" in content
    assert "Hello" in content
    assert "This is a test message." in content


def test_list_sessions(tmp_path: Any) -> None:
    """Test listing all available sessions."""
    project_path = pathlib.Path(tmp_path) / "test_project"
    settings.DOMARKX_PROJECT_PATH = str(project_path)
    (project_path / "sessions").mkdir(parents=True, exist_ok=True)
    session_file = project_path / "sessions" / "test_session.md"
    session_file.write_text("Test Session\nHello")
    sessions = list_sessions()
    assert "test_session.md" in sessions
