"""World Manager for the DAM system."""

import logging
from typing import TYPE_CHECKING

from dam.core.config import Settings
from dam.core.config import settings as global_app_settings

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

    def get_default_world(self) -> "World | None":
        """Get the default world instance."""
        default_name = global_app_settings.DEFAULT_WORLD_NAME
        if default_name:
            return self.get_world(default_name)
        return None

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

    def create_and_register_world(self, world_name: str, app_settings: Settings | None = None) -> "World":
        """Create a new world instance and register it."""
        from dam.core.world import World  # noqa: PLC0415

        current_settings = app_settings or global_app_settings
        logger.info(
            "Attempting to create and register world: %s using settings: %s",
            world_name,
            "provided" if app_settings else "global",
        )
        try:
            world_cfg = current_settings.get_world_config(world_name)
        except ValueError as e:
            logger.error("Failed to get configuration for world '%s': %s", world_name, e)
            raise

        world = World(world_config=world_cfg)

        from dam.plugins.core import CorePlugin  # noqa: PLC0415

        world.add_plugin(CorePlugin())

        world.scheduler.resource_manager = world.resource_manager

        world.logger.info("World '%s' resources populated and scheduler updated.", world.name)

        self.register_world(world)
        return world

    def create_and_register_all_worlds_from_settings(self, app_settings: Settings | None = None) -> list["World"]:
        """Create and register all worlds defined in the application settings."""
        current_settings = app_settings or global_app_settings
        created_worlds: list[World] = []
        world_names = current_settings.get_all_world_names()
        logger.info(
            "Found %d worlds in settings to create and register: %s (using %s settings)",
            len(world_names),
            world_names,
            "provided" if app_settings else "global",
        )

        for name in world_names:
            try:
                world = self.create_and_register_world(name, app_settings=current_settings)
                created_worlds.append(world)
            except Exception:
                logger.exception("Failed to create or register world '%s'", name)
        return created_worlds
