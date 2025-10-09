"""Manages the global state for the CLI, primarily the active world."""

from dam.core.world import World
from dam.core.world import get_world as get_world_from_core


class GlobalState:
    """A simple class to hold global application state."""

    world_name: str | None = None


global_state = GlobalState()


def get_world() -> World | None:
    """
    Get the currently active world instance based on the global state.

    Returns:
        The active World instance or None if no world is set.

    """
    if global_state.world_name:
        return get_world_from_core(global_state.world_name)
    return None
