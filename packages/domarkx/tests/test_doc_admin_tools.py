import os
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from domarkx.tools.doc_admin import (
    rename_session,
    summarize_conversation,
    update_session_metadata,
)

runner = CliRunner()


class TestDocAdminTools(unittest.TestCase):
    @patch("subprocess.run")
    def test_rename_session(self, mock_run):
        with runner.isolated_filesystem() as fs:
            os.makedirs("sessions")
            with open("sessions/test_session.md", "w") as f:
                f.write("test content")
            result = rename_session("test_session", "new_session_name", project_path=fs)
            self.assertEqual(result, "Session 'test_session' renamed to 'new_session_name'.")
            self.assertFalse(os.path.exists("sessions/test_session.md"))
            self.assertTrue(os.path.exists("sessions/new_session_name.md"))

    @patch("subprocess.run")
    def test_update_session_metadata(self, mock_run):
        with runner.isolated_filesystem() as fs:
            os.makedirs("sessions")
            with open("sessions/test_session.md", "w") as f:
                f.write("test content")
            metadata = {"key": "value"}
            result = update_session_metadata("test_session", metadata, project_path=fs)
            self.assertEqual(result, "Metadata updated for session 'test_session'.")
            with open("sessions/test_session.md", "r") as f:
                content = f.read()
                self.assertIn(str(metadata), content)

    @patch("domarkx.tools.doc_admin.create_session")
    @patch("domarkx.tools.doc_admin.send_message")
    def test_summarize_conversation(self, mock_send_message, mock_create_session):
        with runner.isolated_filesystem() as fs:
            os.makedirs("sessions")
            with open("sessions/test_session.md", "w") as f:
                f.write("test content")
            result = summarize_conversation("test_session", project_path=fs)
            self.assertEqual(
                result,
                "Summarization request sent for session 'test_session'. Check the 'summarizer-for-test_session' session for the summary.",
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


if __name__ == "__main__":
    unittest.main()
