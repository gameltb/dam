"""Tests for the document administration tools."""

import pathlib
from typing import Any

from typer.testing import CliRunner

from domarkx.tools.doc_admin import summarize_conversation

runner = CliRunner()


def test_summarize_conversation(mocker: Any) -> None:
    """Verify that the summarize_conversation function correctly sends a summarization request."""
    mock_create_session = mocker.patch("domarkx.tools.doc_admin.create_session")
    mock_send_message = mocker.patch("domarkx.tools.doc_admin.send_message")
    with runner.isolated_filesystem() as fs:
        fs_path = pathlib.Path(fs)
        (fs_path / "sessions").mkdir()
        (fs_path / "sessions" / "test_session.md").write_text("test content")
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
