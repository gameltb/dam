"""Settings for the dam_app plugin."""

from dam.models.config import ConfigComponent, SettingsModel


class AppSettingsModel(SettingsModel):
    """Pydantic model for validating dam_app plugin settings."""

    pass


class AppSettingsComponent(ConfigComponent):
    """ECS component holding the dam_app settings."""

    __tablename__ = "dam_app_config"
