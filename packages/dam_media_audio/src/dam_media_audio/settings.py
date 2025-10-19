"""Settings for the dam_media_audio plugin."""

from dam.models.config import ConfigComponent, SettingsModel


class AudioSettingsModel(SettingsModel):
    """Pydantic model for validating dam_media_audio plugin settings."""

    pass


class AudioSettingsComponent(ConfigComponent):
    """ECS component holding the dam_media_audio settings."""

    __tablename__ = "dam_media_audio_config"
