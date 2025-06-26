import logging

from dam.core.config import WorldConfig
from dam.core.config import settings as global_app_settings
from dam.core.database import DatabaseManager
from dam.core.resources import FileOperationsResource
from dam.core.world import World  # Forward declaration for type hint, or direct import if no circularity
from dam.services.file_storage_service import FileStorageService

logger = logging.getLogger(__name__)


def populate_base_resources(world: World) -> None:
    """
    Populates the given World instance with essential base resources.

    These include:
    - WorldConfig (the world's own configuration object)
    - DatabaseManager
    - FileStorageService
    - FileOperationsResource
    """
    if not world:
        raise ValueError("A valid World instance must be provided.")

    world_config = world.config  # Get the config from the world instance itself

    world.logger.info(f"Populating base resources for World '{world.name}'...")

    # 1. WorldConfig itself as a resource
    world.resource_manager.add_resource(world_config, WorldConfig)
    world.logger.debug(f"Added WorldConfig resource for World '{world.name}'.")

    # 2. DatabaseManager
    db_manager = DatabaseManager(
        world_config=world_config,
        testing_mode=global_app_settings.TESTING_MODE,  # Use global app setting for testing mode
    )
    world.resource_manager.add_resource(db_manager, DatabaseManager)
    # Also store a direct reference if World methods like get_db_session need it without get_resource
    # However, the current get_db_session in World uses get_resource(DatabaseManager) which is good.
    # world.db_manager = db_manager # No longer storing direct .db_manager attribute on World
    world.logger.debug(f"Added DatabaseManager resource for World '{world.name}'.")

    # 3. FileStorageService
    file_storage_svc = FileStorageService(world_config=world_config)
    world.resource_manager.add_resource(file_storage_svc, FileStorageService)
    # world.file_storage_service = file_storage_svc # No longer storing direct .file_storage_service attribute
    world.logger.debug(f"Added FileStorageService resource for World '{world.name}'.")

    # 4. FileOperationsResource
    world.resource_manager.add_resource(FileOperationsResource())
    world.logger.debug(f"Added FileOperationsResource for World '{world.name}'.")

    world.logger.info(
        f"Base resources populated for World '{world.name}'. Current resources: {list(world.resource_manager._resources.keys())}"
    )


__all__ = ["populate_base_resources"]
