import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
from domarkx.tools.doc_admin import (
    rename_session,
    update_session_metadata,
    summarize_conversation,
)
from domarkx.config import settings


class TestDocAdminTools(unittest.TestCase):
    def setUp(self):
        self.test_session_name = "test_session"
        self.test_session_path = os.path.join(settings.project_path, "sessions", f"{self.test_session_name}.md")
        self.sessions_dir = os.path.join(settings.project_path, "sessions")
        os.makedirs(self.sessions_dir, exist_ok=True)
        with open(self.test_session_path, "w") as f:
            f.write("test content")

    def tearDown(self):
        shutil.rmtree(self.sessions_dir)

    @patch("subprocess.run")
    def test_rename_session(self, mock_run):
        new_name = "new_session_name"
        result = rename_session(self.test_session_name, new_name)
        self.assertEqual(result, f"Session '{self.test_session_name}' renamed to '{new_name}'.")
        self.assertFalse(os.path.exists(self.test_session_path))
        self.assertTrue(os.path.exists(os.path.join(self.sessions_dir, f"{new_name}.md")))

    @patch("subprocess.run")
    def test_update_session_metadata(self, mock_run):
        metadata = {"key": "value"}
        result = update_session_metadata(self.test_session_name, metadata)
        self.assertEqual(result, f"Metadata updated for session '{self.test_session_name}'.")
        with open(self.test_session_path, "r") as f:
            content = f.read()
            self.assertIn(str(metadata), content)

    @patch("domarkx.tools.doc_admin.create_session")
    @patch("domarkx.tools.doc_admin.send_message")
    def test_summarize_conversation(self, mock_send_message, mock_create_session):
        result = summarize_conversation(self.test_session_name)
        self.assertEqual(
            result,
            f"Summarization request sent for session '{self.test_session_name}'. Check the 'summarizer-for-{self.test_session_name}' session for the summary.",
        )
        mock_create_session.assert_called_once_with(
            "ConversationSummarizer",
            f"summarizer-for-{self.test_session_name}",
            {"session_to_summarize": self.test_session_name},
        )
        mock_send_message.assert_called_once_with(
            f"summarizer-for-{self.test_session_name}",
            f"Please summarize the conversation in session '{self.test_session_name}'.",
        )


if __name__ == "__main__":
    unittest.main()
