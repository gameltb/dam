import os

from domarkx.config import settings
from domarkx.tools.session_management import (
    create_session,
    get_messages,
    list_sessions,
    send_message,
)


def test_create_session(tmp_path):
    project_path = tmp_path / "test_project"
    os.makedirs(project_path / "sessions")
    os.makedirs(project_path / "templates")
    # 写入模板文件，使用 domarkx 宏格式
    with open(project_path / "templates" / "default.md", "w") as f:
        f.write("# [@session_name](domarkx://session_name)\n\n[@user_prompt](domarkx://user_prompt)")
    settings.DOMARKX_PROJECT_PATH = str(project_path)
    result = create_session("default", "test_session", {"session_name": "Test Session", "user_prompt": "Hello"})
    assert result == "Session 'test_session' created from template 'default'."
    session_file = project_path / "sessions" / "test_session.md"
    assert session_file.exists()
    with open(session_file, "r") as f:
        content = f.read()
        assert "Test Session" in content
        assert "Hello" in content


def test_send_message(tmp_path):
    project_path = tmp_path / "test_project"
    settings.DOMARKX_PROJECT_PATH = str(project_path)
    # 保证 session 文件已存在
    os.makedirs(project_path / "sessions", exist_ok=True)
    session_file = project_path / "sessions" / "test_session.md"
    with open(session_file, "w") as f:
        f.write("Test Session\nHello")
    result = send_message("test_session", "This is a test message.")
    assert result == "Message sent to session 'test_session'."
    with open(session_file, "r") as f:
        content = f.read()
        assert "This is a test message." in content


def test_get_messages(tmp_path):
    project_path = tmp_path / "test_project"
    settings.DOMARKX_PROJECT_PATH = str(project_path)
    os.makedirs(project_path / "sessions", exist_ok=True)
    session_file = project_path / "sessions" / "test_session.md"
    with open(session_file, "w") as f:
        f.write("Test Session\nHello\nThis is a test message.")
    content = get_messages("test_session")
    assert "Test Session" in content
    assert "Hello" in content
    assert "This is a test message." in content


def test_list_sessions(tmp_path):
    project_path = tmp_path / "test_project"
    settings.DOMARKX_PROJECT_PATH = str(project_path)
    os.makedirs(project_path / "sessions", exist_ok=True)
    session_file = project_path / "sessions" / "test_session.md"
    with open(session_file, "w") as f:
        f.write("Test Session\nHello")
    sessions = list_sessions()
    assert "test_session.md" in sessions
