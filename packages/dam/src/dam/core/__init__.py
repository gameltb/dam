# Core exports for the DAM system

from .config import Settings, WorldConfig, settings
from .contexts import ContextProvider
from .database import Base, DatabaseManager
from .events import BaseEvent
from .exceptions import EventHandlingError, StageExecutionError
from .resources import ResourceManager, ResourceNotFoundError
from .stages import SystemStage
from .systems import WorldScheduler, system
from .transaction_manager import TransactionManager
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
    "settings",
    # Contexts
    "ContextProvider",
    # Database
    "DatabaseManager",
    "Base",
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
    # Systems
    "WorldScheduler",
    "system",
    # Transaction
    "TransactionManager",
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
