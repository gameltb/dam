"""Defines the Pydantic models for the dam-media-transcode plugin settings."""

from pathlib import Path

from dam.core.config import PluginSettings


class TranscodePluginSettings(PluginSettings):
    """Settings for the dam-media-transcode plugin."""

    TRANSCODING_TEMP_DIR: Path
