"""Settings for the dam_media_transcode plugin."""

from pathlib import Path

from dam.models.config import ConfigComponent, SettingsModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column


class TranscodeSettingsModel(SettingsModel):
    """Pydantic model for validating dam_media_transcode plugin settings."""

    TRANSCODING_TEMP_DIR: Path


class TranscodeSettingsComponent(ConfigComponent):
    """ECS component holding the dam_media_transcode settings."""

    __tablename__ = "transcode_settings"
    TRANSCODING_TEMP_DIR: Mapped[str] = mapped_column(String, nullable=False)
