import os
import shutil
import subprocess
from domarkx.config import settings
from domarkx.tools.session_management import (
    create_session,
    send_message,
    get_messages,
    list_sessions,
)


def setup_module(module):
    if os.path.exists("test_project"):
        shutil.rmtree("test_project")
    os.makedirs("test_project/sessions")
    os.makedirs("test_project/templates")
    with open("test_project/templates/default.md", "w") as f:
        f.write("# {session_name}\n\n{user_prompt}")
    subprocess.run(["git", "init"], cwd="test_project")
    settings.DOMARKX_PROJECT_PATH = "test_project"


def teardown_module(module):
    shutil.rmtree("test_project")


def test_create_session():
    result = create_session("default", "test_session", {"session_name": "Test Session", "user_prompt": "Hello"})
    assert result == "Session 'test_session' created from template 'default'."
    assert os.path.exists("test_project/sessions/test_session.md")
    with open("test_project/sessions/test_session.md", "r") as f:
        content = f.read()
        assert "Test Session" in content
        assert "Hello" in content


def test_send_message():
    result = send_message("test_session", "This is a test message.")
    assert result == "Message sent to session 'test_session'."
    with open("test_project/sessions/test_session.md", "r") as f:
        content = f.read()
        assert "This is a test message." in content


def test_get_messages():
    content = get_messages("test_session")
    assert "Test Session" in content
    assert "Hello" in content
    assert "This is a test message." in content


def test_list_sessions():
    sessions = list_sessions()
    assert "test_session.md" in sessions
