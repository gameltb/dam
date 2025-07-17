import os
import subprocess
from domarkx.config import settings
from domarkx.tools.session_management import create_session, send_message


from domarkx.tools.tool_decorator import tool_handler


@tool_handler()
def rename_session(old_name: str, new_name: str) -> str:
    """
    Rename a session file in the sessions directory and update git tracking.

    Args:
        old_name (str): The current name of the session file (without .md extension).
        new_name (str): The new name of the session file (without .md extension).

    Returns:
        str: Success message indicating the session was renamed.

    Raises:
        FileNotFoundError: If the old session file does not exist.
    """
    old_path = os.path.join(settings.project_path, "sessions", f"{old_name}.md")
    new_path = os.path.join(settings.project_path, "sessions", f"{new_name}.md")

    if not os.path.exists(old_path):
        raise FileNotFoundError(f"Session not found: {old_path}")

    os.rename(old_path, new_path)

    # Add to git
    subprocess.run(["git", "add", new_path], cwd=settings.project_path)
    subprocess.run(["git", "rm", old_path], cwd=settings.project_path)
    subprocess.run(
        ["git", "commit", "-m", f"Rename session {old_name} to {new_name}"],
        cwd=settings.project_path,
    )

    return f"Session '{old_name}' renamed to '{new_name}'."


@tool_handler()
def update_session_metadata(session_name: str, metadata: dict) -> str:
    """
    Update the metadata block in a session file. Appends metadata as a comment for now.

    Args:
        session_name (str): The name of the session file (without .md extension).
        metadata (dict): A dictionary of metadata to update.

    Returns:
        str: Success message indicating metadata was updated.

    Raises:
        FileNotFoundError: If the session file does not exist.
    """
    session_path = os.path.join(settings.project_path, "sessions", f"{session_name}.md")

    if not os.path.exists(session_path):
        raise FileNotFoundError(f"Session not found: {session_path}")

    # This is a simplified implementation. A more robust solution would parse the
    # markdown and update the metadata block.
    with open(session_path, "r+") as f:
        content = f.read()
        # This is a placeholder for a more robust metadata update logic.
        # For now, we'll just append the metadata as a comment.
        f.write(f"\n\n<!-- METADATA: {metadata} -->")

    # Add to git
    subprocess.run(["git", "add", session_path], cwd=settings.project_path)
    subprocess.run(
        ["git", "commit", "-m", f"Update metadata for session {session_name}"],
        cwd=settings.project_path,
    )

    return f"Metadata updated for session '{session_name}'."


@tool_handler()
def summarize_conversation(session_name: str) -> str:
    """
    Summarize the conversation in a session by delegating to the ConversationSummarizer agent.

    Args:
        session_name (str): The name of the session to summarize.

    Returns:
        str: Message indicating the summarization request was sent.
    """
    summarizer_session_name = f"summarizer-for-{session_name}"
    create_session(
        "ConversationSummarizer",
        summarizer_session_name,
        {"session_to_summarize": session_name},
    )
    send_message(
        summarizer_session_name,
        f"Please summarize the conversation in session '{session_name}'.",
    )
    # In a real implementation, we would wait for the summarizer to finish
    # and get the result. For now, we'll just return a message.
    return f"Summarization request sent for session '{session_name}'. Check the '{summarizer_session_name}' session for the summary."
