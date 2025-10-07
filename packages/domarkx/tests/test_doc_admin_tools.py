import os
from typing import Any

from typer.testing import CliRunner

from domarkx.tools.doc_admin import (
    rename_session,
    summarize_conversation,
    update_session_metadata,
)

runner = CliRunner()


def test_rename_session(mocker: Any) -> None:
    """Verify that the rename_session function correctly renames a session file."""
    mocker.patch("subprocess.run")
    with runner.isolated_filesystem() as fs:
        os.makedirs("sessions")
        with open(os.path.join(fs, "sessions/test_session.md"), "w") as f:
            f.write("test content")
        result = rename_session("test_session", "new_session_name", project_path=fs)
        assert result == "Session 'test_session' renamed to 'new_session_name'."
        assert not os.path.exists(os.path.join(fs, "sessions/test_session.md"))
        assert os.path.exists(os.path.join(fs, "sessions/new_session_name.md"))


def test_update_session_metadata(mocker: Any) -> None:
    """Verify that the update_session_metadata function correctly updates the session metadata."""
    mocker.patch("subprocess.run")
    with runner.isolated_filesystem() as fs:
        os.makedirs("sessions")
        with open(os.path.join(fs, "sessions/test_session.md"), "w") as f:
            f.write("test content")
        metadata = {"key": "value"}
        result = update_session_metadata("test_session", metadata, project_path=fs)
        assert result == "Metadata updated for session 'test_session'."
        with open(os.path.join(fs, "sessions/test_session.md")) as f:
            content = f.read()
            assert str(metadata) in content


def test_summarize_conversation(mocker: Any) -> None:
    """Verify that the summarize_conversation function correctly sends a summarization request."""
    mock_create_session = mocker.patch("domarkx.tools.doc_admin.create_session")
    mock_send_message = mocker.patch("domarkx.tools.doc_admin.send_message")
    with runner.isolated_filesystem() as fs:
        os.makedirs("sessions")
        with open(os.path.join(fs, "sessions/test_session.md"), "w") as f:
            f.write("test content")
        result = summarize_conversation("test_session", project_path=fs)
        assert (
            result
            == "Summarization request sent for session 'test_session'. Check the 'summarizer-for-test_session' session for the summary."
        )
        mock_create_session.assert_called_once_with(
            "ConversationSummarizer",
            "summarizer-for-test_session",
            {"session_to_summarize": "test_session"},
            project_path=fs,
        )
        mock_send_message.assert_called_once_with(
            "summarizer-for-test_session",
            "Please summarize the conversation in session 'test_session'.",
            project_path=fs,
        )
