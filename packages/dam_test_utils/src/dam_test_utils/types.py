"""Types for DAM tests."""

from collections.abc import Awaitable, Callable

from dam.core.world import World
from dam.models.config import ConfigComponent

WorldFactory = Callable[[str, list[ConfigComponent]], Awaitable[World]]
