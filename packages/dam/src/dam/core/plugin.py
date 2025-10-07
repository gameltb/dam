from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dam.core.world import World


class Plugin(Protocol):
    """A protocol for plugins that can be added to a World."""

    def build(self, world: "World") -> None:
        """
        Builds the plugin, adding systems, resources, etc. to the world.

        Args:
            world: The world to build the plugin into.

        """
        ...
