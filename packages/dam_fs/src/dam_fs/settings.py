"""Settings for the dam_fs plugin."""

from pathlib import Path

from dam.models.config import ConfigComponent, SettingsModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column


class FsSettingsModel(SettingsModel):
    """Pydantic model for validating dam_fs plugin settings."""

    ASSET_STORAGE_PATH: Path


class FsSettingsComponent(ConfigComponent):
    """ECS component holding the dam_fs settings."""

    __tablename__ = "fs_settings"
    ASSET_STORAGE_PATH: Mapped[str] = mapped_column(String, nullable=False)
