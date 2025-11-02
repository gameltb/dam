"""Basic tests for the Llama Server Launcher."""

from llama_server_launcher.main import LlamaServerLauncher


def test_import() -> None:
    """Test that the application can be imported."""
    assert LlamaServerLauncher is not None
