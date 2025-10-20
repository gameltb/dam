"""Fundamental building blocks of the DAM ECS framework."""
# Core exports for the DAM system

from .database import Base, DatabaseManager, DBSession
from .exceptions import EventHandlingError, StageExecutionError
from .resources import ResourceManager, ResourceNotFoundError
from .stages import SystemStage
from .systems import WorldScheduler, system
from .world import World

__all__ = [
    "Base",
    "DBSession",
    "DatabaseManager",
    "EventHandlingError",
    "ResourceManager",
    "ResourceNotFoundError",
    "StageExecutionError",
    "SystemStage",
    "World",
    "WorldScheduler",
    "system",
]
