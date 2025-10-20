"""Settings for the dam_semantic plugin."""

from dam.models.config import ConfigComponent, SettingsModel


class SemanticSettingsModel(SettingsModel):
    """Pydantic model for validating dam_semantic plugin settings."""

    pass


class SemanticSettingsComponent(ConfigComponent):
    """ECS component holding the dam_semantic settings."""

    __tablename__ = "dam_semantic_config"
