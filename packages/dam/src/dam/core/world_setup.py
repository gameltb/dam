import logging

# --- Core System Registration (merged from world_registrar.py) ---
from typing import TYPE_CHECKING

from dam_fs.resources.file_storage_resource import FileStorageResource

from dam.core.config import settings as global_app_settings  # WorldConfig no longer needed here directly
from dam.core.database import DatabaseManager

# Moved these imports to the top of the file, outside of the TYPE_CHECKING block
# to resolve E402 errors, as they are needed at runtime for registration.
from dam.core.resources import FileOperationsResource, HashingFunctionsResource
from dam.core.world import World

if TYPE_CHECKING:
    # World is already imported at the top of this file for initialize_world_resources
    # No runtime imports needed here if they are already above.
    pass
logger = logging.getLogger(__name__)


def initialize_world_resources(world: World) -> None:
    """
    Populates the given World instance's ResourceManager with essential base resources.

    These include:
    - WorldConfig (the world's own configuration object)
    - DatabaseManager
    - FileStorageResource
    - FileOperationsResource
    """
    if not world:
        raise ValueError("A valid World instance must be provided.")

    world_config = world.config  # Get config from the world instance
    resource_manager = world.resource_manager  # Use the world's existing resource manager
    world_name = world_config.name

    world.logger.info(f"Populating base resources for World '{world_name}'...")

    # 1. WorldConfig itself as a resource
    resource_manager.add_resource(world_config, world_config.__class__)  # Use world_config.__class__ for type
    world.logger.debug(f"Added WorldConfig resource for World '{world_name}'.")

    # 2. DatabaseManager
    db_manager = DatabaseManager(
        world_config=world_config,
        testing_mode=global_app_settings.TESTING_MODE,  # Use global app setting for testing mode
    )
    resource_manager.add_resource(db_manager, DatabaseManager)
    world.logger.debug(f"Added DatabaseManager resource for World '{world_name}'.")

    # 3. FileStorageResource
    file_storage_svc = FileStorageResource(world_config=world_config)
    resource_manager.add_resource(file_storage_svc, FileStorageResource)
    world.logger.debug(f"Added FileStorageResource resource for World '{world_name}'.")

    # 4. FileOperationsResource
    resource_manager.add_resource(FileOperationsResource())
    world.logger.debug(f"Added FileOperationsResource for World '{world_name}'.")

    # 5. HashingFunctionsResource
    resource_manager.add_resource(HashingFunctionsResource())
    world.logger.debug(f"Added HashingFunctionsResource for World '{world_name}'.")

    world.logger.info(
        f"Base resources populated for World '{world_name}'. Current resources: {list(resource_manager._resources.keys())}"
    )
    # No return value as it modifies the world's resource_manager in-place


__all__ = ["initialize_world_resources", "register_core_systems"]


def register_core_systems(world_instance: "World") -> None:
    """
    Registers all standard, core systems for a given world instance.
    This ensures consistency in system registration across different application entry points.
    """
    logger.info(f"Core system registration complete for world: {world_instance.name}")
