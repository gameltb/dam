"""World Manager for the DAM system."""

import logging
from typing import TYPE_CHECKING

from dam.core.config import Config
from dam.core.plugin import discover_plugin_settings

if TYPE_CHECKING:
    from dam.core.world import World


logger = logging.getLogger(__name__)


class WorldManager:
    """Manages the lifecycle and registry of all World instances."""

    def __init__(self) -> None:
        """Initialize the WorldManager."""
        self._worlds: dict[str, World] = {}

    def register_world(self, world_instance: "World") -> None:
        """Register a world instance."""
        from dam.core.world import World  # noqa: PLC0415

        if not isinstance(world_instance, World):
            raise TypeError("Can only register instances of World.")
        if world_instance.name in self._worlds:
            logger.warning("World with name '%s' is already registered. Overwriting.", world_instance.name)
        self._worlds[world_instance.name] = world_instance
        logger.info("World '%s' registered.", world_instance.name)

    def get_world(self, world_name: str) -> "World | None":
        """Get a world instance by name."""
        return self._worlds.get(world_name)

    def get_all_registered_worlds(self) -> list["World"]:
        """Return a list of all registered world instances."""
        return list(self._worlds.values())

    def unregister_world(self, world_name: str) -> bool:
        """Remove a world instance from the registry."""
        if world_name in self._worlds:
            del self._worlds[world_name]
            logger.info("World '%s' unregistered.", world_name)
            return True
        logger.warning("Attempted to unregister World '%s', but it was not found.", world_name)
        return False

    def clear_world_registry(self) -> None:
        """Clear all worlds from the registry."""
        count = len(self._worlds)
        self._worlds.clear()
        logger.info("Cleared %d worlds from the registry.", count)


def create_world(world_name: str, config: Config) -> "World":
    """
    Create a new World instance from a configuration definition.

    This function is the central point for instantiating a world. It handles:
    - Retrieving the world's configuration definition.
    - Discovering all available plugin settings classes.
    - Validating the plugin settings from the config against the discovered classes.
    - Instantiating the World object.
    - Adding the validated plugin settings as ECS resources.
    - Adding the core plugin.

    Args:
        world_name: The name of the world to create.
        config: The loaded application configuration.

    Returns:
        A newly created (but not yet registered) World instance.

    Raises:
        ValueError: If the world is not defined in the configuration.

    """
    from dam.core.world import World  # noqa: PLC0415
    from dam.plugins.core import CorePlugin  # noqa: PLC0415

    world_def = config.worlds.get(world_name)
    if not world_def:
        raise ValueError(f"World '{world_name}' not defined in the configuration.")

    logger.info("Creating world '%s' from definition.", world_name)

    # Instantiate the world with its core definition
    world = World(name=world_name, definition=world_def)

    # Discover and process plugin settings
    discovered_settings = discover_plugin_settings()
    raw_settings = world_def.plugin_settings

    for name, settings_class in discovered_settings.items():
        settings_data = raw_settings.get(name, {})
        try:
            # Validate the settings and add as a resource
            validated_settings = settings_class.model_validate(settings_data)
            world.add_resource(validated_settings)
            logger.debug("Validated and added settings for plugin '%s' to world '%s'.", name, world_name)
        except Exception:
            logger.exception("Failed to validate settings for plugin '%s' in world '%s'.", name, world_name)
            raise

    # Add the core plugin, which populates essential resources like the DB manager
    world.add_plugin(CorePlugin())

    logger.info("World '%s' created and core plugin loaded.", world_name)
    return world


# Global world manager instance
world_manager = WorldManager()
