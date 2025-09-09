from typing import Optional

from dam.core.world import get_world as get_world_from_core


class GlobalState:
    world_name: Optional[str] = None


global_state = GlobalState()


def get_world():
    if global_state.world_name:
        return get_world_from_core(global_state.world_name)
    return None
