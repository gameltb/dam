"""Settings for the dam_source plugin."""

from dam.models.config import ConfigComponent, SettingsModel


class SourceSettingsModel(SettingsModel):
    """Pydantic model for validating dam_source plugin settings."""

    pass


class SourceSettingsComponent(ConfigComponent):
    """ECS component holding the dam_source settings."""

    __tablename__ = "dam_source_config"
