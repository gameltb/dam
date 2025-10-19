"""Settings for the dam_sire plugin."""

from dam.models.config import ConfigComponent, SettingsModel


class SireSettingsModel(SettingsModel):
    """Pydantic model for validating dam_sire plugin settings."""

    pass


class SireSettingsComponent(ConfigComponent):
    """ECS component holding the dam_sire settings."""

    __tablename__ = "dam_sire_config"
