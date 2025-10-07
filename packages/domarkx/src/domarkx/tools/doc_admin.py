"""Administrative tools for managing documentation sessions."""

from domarkx.tools.session_management import (
    archive_session,
    create_session,
    list_sessions,
    send_message,
)
from domarkx.tools.tool_factory import tool_handler


@tool_handler()
def summarize_conversation(session_name: str, project_path: str | None = None) -> str:
    """
    Summarize a conversation in a session file.

    Args:
        session_name (str): The name of the session file (without .md extension).
        project_path (str, optional): The path to the project. Defaults to None.

    Returns:
        str: Success message.

    """
    create_session(
        "ConversationSummarizer",
        f"summarizer-for-{session_name}",
        {"session_to_summarize": session_name},
        project_path=project_path,
    )
    send_message(
        f"summarizer-for-{session_name}",
        f"Please summarize the conversation in session '{session_name}'.",
        project_path=project_path,
    )
    return f"Summarization request sent for session '{session_name}'. Check the 'summarizer-for-{session_name}' session for the summary."


__all__ = [
    "archive_session",
    "create_session",
    "list_sessions",
    "send_message",
    "summarize_conversation",
]
