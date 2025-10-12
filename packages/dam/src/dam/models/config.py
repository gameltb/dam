"""Base classes for configuration components and models in the DAM system."""

from pydantic import BaseModel

from dam.models.core.base_component import BaseComponent as DamBaseComponent


class SettingsModel(BaseModel):
    """
    A base class for Pydantic models used to deserialize settings from TOML.

    Plugins should create a subclass of this model to define their configuration
    schema, which will be used by the configuration loader to validate the
    `[worlds.<world_name>.plugin_settings.<plugin_name>]` table in `dam.toml`.
    """

    pass


class ConfigComponent(DamBaseComponent):
    """
    An abstract base class for ECS components that store plugin configurations.

    Plugins should create a subclass of this component to hold their settings
    within a world. The data for this component will be populated from a
    corresponding `SettingsModel` after being loaded and validated from `dam.toml`.

    This component is designed to be persisted in the database, linked to an entity,
    allowing configurations to be versioned and managed as part of the world's state.
    """

    __abstract__ = True
