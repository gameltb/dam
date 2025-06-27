import logging

from dam.core.config import WorldConfig
from dam.core.config import settings as global_app_settings
from dam.core.database import DatabaseManager
from dam.core.resources import FileOperationsResource, ResourceManager
from dam.services.file_storage_service import FileStorageService

logger = logging.getLogger(__name__)


def initialize_world_resources(world_config: WorldConfig) -> ResourceManager:
    """
    Initializes and returns a ResourceManager populated with essential base resources
    for a world, based on the provided WorldConfig.

    These include:
    - WorldConfig (the world's own configuration object)
    - DatabaseManager
    - FileStorageService
    - FileOperationsResource
    """
    if not world_config:
        raise ValueError("A valid WorldConfig instance must be provided.")

    resource_manager = ResourceManager()
    world_name = world_config.name  # Get name from config for logging

    logger.info(f"Initializing base resources for World '{world_name}'...")

    # 1. WorldConfig itself as a resource
    resource_manager.add_resource(world_config, WorldConfig)
    logger.debug(f"Added WorldConfig resource for World '{world_name}'.")

    # 2. DatabaseManager
    db_manager = DatabaseManager(
        world_config=world_config,
        testing_mode=global_app_settings.TESTING_MODE,  # Use global app setting for testing mode
    )
    resource_manager.add_resource(db_manager, DatabaseManager)
    logger.debug(f"Added DatabaseManager resource for World '{world_name}'.")

    # 3. FileStorageService
    file_storage_svc = FileStorageService(world_config=world_config)
    resource_manager.add_resource(file_storage_svc, FileStorageService)
    logger.debug(f"Added FileStorageService resource for World '{world_name}'.")

    # 4. FileOperationsResource
    resource_manager.add_resource(FileOperationsResource())
    logger.debug(f"Added FileOperationsResource for World '{world_name}'.")

    logger.info(
        f"Base resources initialized for World '{world_name}'. Current resources: {list(resource_manager._resources.keys())}"
    )
    return resource_manager


__all__ = ["initialize_world_resources"]
