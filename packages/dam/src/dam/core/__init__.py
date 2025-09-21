# Core exports for the DAM system

from .config import Settings, WorldConfig, settings
from .database import Base, DatabaseManager  # Removed get_async_engine, etc.
from .events import BaseEvent  # Add specific events if they are broadly used, or import from events module directly
from .exceptions import EventHandlingError, StageExecutionError  # ResourceNotFoundError comes from .resources
from .resources import ResourceManager, ResourceNotFoundError  # Import ResourceNotFoundError from here
from .stages import SystemStage
from .systems import WorldScheduler, system  # SYSTEM_METADATA might be too internal
from .world import (
    World,
    clear_world_registry,
    create_and_register_all_worlds_from_settings,
    create_and_register_world,
    get_all_registered_worlds,
    get_default_world,
    get_world,
    register_world,  # Added missing register_world
    unregister_world,
)

__all__ = [
    # Config
    "Settings",
    "WorldConfig",
    "settings",  # Global settings instance
    # Database
    "DatabaseManager",
    "Base",  # SQLAlchemy declarative base
    # Events
    "BaseEvent",
    # Exceptions
    "EventHandlingError",
    "StageExecutionError",
    "ResourceNotFoundError",
    # Resources
    "ResourceManager",
    # Stages
    "SystemStage",
    # Systems Decorators & Scheduler
    "WorldScheduler",  # Class for managing system execution per world
    "system",  # Decorator for stage-based systems
    # World
    "World",
    "get_world",
    "get_default_world",
    "create_and_register_world",
    "create_and_register_all_worlds_from_settings",
    "clear_world_registry",
    "unregister_world",
    "get_all_registered_worlds",
    "register_world",
]
