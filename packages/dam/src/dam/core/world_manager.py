"""World Manager for the DAM system."""

import logging
from typing import TYPE_CHECKING

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
