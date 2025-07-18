import os
import shutil
import subprocess
from domarkx.config import settings
from domarkx.tools.tool_decorator import tool_handler

from domarkx.session import MacroExpander


@tool_handler()
def create_session(template_name: str, session_name: str, parameters: dict):
    """
    Create a new session file from a template and parameters.

    Args:
        template_name (str): The name of the template (without .md extension).
        session_name (str): The name of the session to create (will be used as filename).
        parameters (dict): Dictionary of parameters to expand in the template. Must include any macros used in the template.

    Returns:
        str: Success message.

    Raises:
        FileNotFoundError: If the template file does not exist.

    Example:
        >>> create_session("default", "test_session", {"session_name": "Test Session", "user_prompt": "Hello"})
        "Session 'test_session' created from template 'default'."

    The session file will be created in the sessions directory and committed to git.
    """
    template_path = os.path.join(settings.project_path, "templates", f"{template_name}.md")
    session_path = os.path.join(settings.project_path, "sessions", f"{session_name}.md")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r") as f:
        template_content = f.read()

    expander = MacroExpander(base_dir=os.path.join(settings.project_path, "templates"))

    # 兼容 domarkx 设计文档中的宏格式（domarkx://...），参数展开不再用 {param}，而是通过 MacroExpander 的宏机制
    # parameters 只作为宏扩展的参数字典传递，expander.expand 会自动处理 domarkx:// 语法
    expanded_content = expander.expand(template_content, parameters={"session_name": session_name, **parameters})

    with open(session_path, "w") as f:
        f.write(expanded_content)

    # Add to git
    subprocess.run(["git", "add", session_path], cwd=settings.project_path)
    subprocess.run(
        ["git", "commit", "-m", f"Create session {session_name}"],
        cwd=settings.project_path,
    )

    return f"Session '{session_name}' created from template '{template_name}'."


@tool_handler()
def send_message(session_name: str, message: str):
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
    session_path = os.path.join(settings.project_path, "sessions", f"{session_name}.md")

    if not os.path.exists(session_path):
        raise FileNotFoundError(f"Session not found: {session_path}")

    with open(session_path, "a") as f:
        f.write(f"\n\n{message}")

    # Add to git
    subprocess.run(["git", "add", session_path], cwd=settings.project_path)
    subprocess.run(
        ["git", "commit", "-m", f"Send message to session {session_name}"],
        cwd=settings.project_path,
    )

    return f"Message sent to session '{session_name}'."


@tool_handler()
def get_messages(session_name: str):
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
    session_path = os.path.join(settings.project_path, "sessions", f"{session_name}.md")

    if not os.path.exists(session_path):
        raise FileNotFoundError(f"Session not found: {session_path}")

    with open(session_path, "r") as f:
        content = f.read()

    return content


@tool_handler()
def list_sessions():
    """
    List all available session files in the sessions directory.

    Returns:
        str: Newline-separated list of session filenames (ending with .md).

    Example:
        >>> list_sessions()
        "test_session.md\nanother_session.md"
    """
    sessions_path = os.path.join(settings.project_path, "sessions")
    sessions = [f for f in os.listdir(sessions_path) if f.endswith(".md")]
    return "\n".join(sessions)
