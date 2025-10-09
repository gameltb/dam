"""Fundamental building blocks of the DAM ECS framework."""
# Core exports for the DAM system

from .config import Settings, WorldConfig, settings
from .database import Base, DatabaseManager
from .exceptions import EventHandlingError, StageExecutionError
from .resources import ResourceManager, ResourceNotFoundError
from .stages import SystemStage
from .systems import WorldScheduler, system
from .world import World

__all__ = [
    "Base",
    # Database
    "DatabaseManager",
    # Exceptions
    "EventHandlingError",
    # Resources
    "ResourceManager",
    "ResourceNotFoundError",
    # Config
    "Settings",
    "StageExecutionError",
    # Stages
    "SystemStage",
    # World
    "World",
    "WorldConfig",
    # Systems
    "WorldScheduler",
    "settings",
    "system",
]
