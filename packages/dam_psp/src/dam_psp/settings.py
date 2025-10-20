"""Settings for the dam_psp plugin."""

from dam.models.config import ConfigComponent, SettingsModel


class PspSettingsModel(SettingsModel):
    """Pydantic model for validating dam_psp plugin settings."""

    pass


class PspSettingsComponent(ConfigComponent):
    """ECS component holding the dam_psp settings."""

    __tablename__ = "dam_psp_config"
