"""Settings for the FsPlugin."""

from pathlib import Path

from dam.core.config import PluginSettings


class FsPluginSettings(PluginSettings):
    """Settings for the FsPlugin."""

    storage_path: Path
