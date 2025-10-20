"""Settings for the dam_archive plugin."""

from dam.models.config import ConfigComponent, SettingsModel


class ArchiveSettingsModel(SettingsModel):
    """Pydantic model for validating dam_archive plugin settings."""

    pass


class ArchiveSettingsComponent(ConfigComponent):
    """ECS component holding the dam_archive settings."""

    __tablename__ = "dam_archive_config"
