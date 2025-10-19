"""Settings for the dam_media_image plugin."""

from dam.models.config import ConfigComponent, SettingsModel


class ImageSettingsModel(SettingsModel):
    """Pydantic model for validating dam_media_image plugin settings."""

    pass


class ImageSettingsComponent(ConfigComponent):
    """ECS component holding the dam_media_image settings."""

    __tablename__ = "dam_media_image_config"
