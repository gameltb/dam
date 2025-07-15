import os
import shutil
import subprocess
from domarkx.config import settings


from domarkx.macro_expander import MacroExpander


def create_session(template_name: str, session_name: str, **kwargs):
    """
    Creates a new session from a template.
    """
    template_path = os.path.join(settings.project_path, "templates", f"{template_name}.md")
    session_path = os.path.join(settings.project_path, "sessions", f"{session_name}.md")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r") as f:
        template_content = f.read()

    expander = MacroExpander(base_dir=os.path.join(settings.project_path, "templates"))

    expanded_content = expander.expand(template_content, parameters={"session_name": session_name, **kwargs})

    with open(session_path, "w") as f:
        f.write(expanded_content)

    # Add to git
    subprocess.run(["git", "add", session_path], cwd=settings.project_path)
    subprocess.run(
        ["git", "commit", "-m", f"Create session {session_name}"],
        cwd=settings.project_path,
    )

    return f"Session '{session_name}' created from template '{template_name}'."


def send_message(session_name: str, message: str):
    """
    Sends a message to a session.
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


def get_messages(session_name: str):
    """
    Gets all messages from a session.
    """
    session_path = os.path.join(settings.project_path, "sessions", f"{session_name}.md")

    if not os.path.exists(session_path):
        raise FileNotFoundError(f"Session not found: {session_path}")

    with open(session_path, "r") as f:
        content = f.read()

    return content


def list_sessions():
    """
    Lists all available sessions.
    """
    sessions_path = os.path.join(settings.project_path, "sessions")
    sessions = [
        f for f in os.listdir(sessions_path) if f.endswith(".md")
    ]
    return "\n".join(sessions)
