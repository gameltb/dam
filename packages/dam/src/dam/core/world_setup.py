import logging

# --- Core System Registration (merged from world_registrar.py) ---
from typing import TYPE_CHECKING

from dam.core.config import settings as global_app_settings  # WorldConfig no longer needed here directly
from dam.core.database import DatabaseManager

# Moved these imports to the top of the file, outside of the TYPE_CHECKING block
# to resolve E402 errors, as they are needed at runtime for registration.
from dam.core.events import (
    AssetFileIngestionRequested,
    AssetReferenceIngestionRequested,
    FindEntityByHashQuery,
    SemanticSearchQuery,
)
from dam.core.model_manager import ModelExecutionManager  # Added ModelExecutionManager
from dam.core.resources import FileOperationsResource, HashingServiceResource
from dam.core.stages import SystemStage
from dam.core.world import World
from dam.resources.file_storage_resource import FileStorageResource

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
    - FileStorageService
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

    # 5. HashingServiceResource
    resource_manager.add_resource(HashingServiceResource())
    world.logger.debug(f"Added HashingServiceResource for World '{world_name}'.")

    # 6. ModelExecutionManager
    # Use the global singleton instance.
    # The ResourceManager's add_resource will handle if this type is already registered,
    # potentially replacing it or warning. For a global singleton, we want all worlds
    # to use the same instance.
    from dam.core.global_resources import get_global_model_execution_manager

    global_model_manager = get_global_model_execution_manager()
    resource_manager.add_resource(global_model_manager, ModelExecutionManager)
    world.logger.debug(f"Added global ModelExecutionManager instance as a resource for World '{world_name}'.")

    # TaggingService is no longer a resource class.
    # Systems needing tagging functions will import them from the tagging_service module
    # and use the injected ModelExecutionManager.

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
    logger.info(f"Registering core systems for world: {world_instance.name}")

    # Metadata Systems
    try:
        from dam.systems.metadata_systems import extract_metadata_on_asset_ingested

        world_instance.register_system(extract_metadata_on_asset_ingested, stage=SystemStage.METADATA_EXTRACTION)
        logger.debug("Registered system: extract_metadata_on_asset_ingested for stage METADATA_EXTRACTION")
    except ImportError as e:
        logger.warning(
            f"Could not import or register extract_metadata_on_asset_ingested, likely due to missing optional dependencies: {e}"
        )

    # Asset Lifecycle Systems (Event-based)
    try:
        from dam.systems.asset_lifecycle_systems import handle_asset_file_ingestion_request

        world_instance.register_system(
            handle_asset_file_ingestion_request,
            event_type=AssetFileIngestionRequested,
        )
        logger.debug("Registered system: handle_asset_file_ingestion_request for event AssetFileIngestionRequested")
    except ImportError as e:
        logger.warning(
            f"Could not import or register handle_asset_file_ingestion_request, likely due to missing optional dependencies: {e}"
        )

    try:
        from dam.systems.asset_lifecycle_systems import handle_asset_reference_ingestion_request

        world_instance.register_system(
            handle_asset_reference_ingestion_request,
            event_type=AssetReferenceIngestionRequested,
        )
        logger.debug(
            "Registered system: handle_asset_reference_ingestion_request for event AssetReferenceIngestionRequested"
        )
    except ImportError as e:
        logger.warning(
            f"Could not import or register handle_asset_reference_ingestion_request, likely due to missing optional dependencies: {e}"
        )

    try:
        from dam.systems.asset_lifecycle_systems import handle_find_entity_by_hash_query

        world_instance.register_system(handle_find_entity_by_hash_query, event_type=FindEntityByHashQuery)
        logger.debug("Registered system: handle_find_entity_by_hash_query for event FindEntityByHashQuery")
    except ImportError as e:
        logger.warning(
            f"Could not import or register handle_find_entity_by_hash_query, likely due to missing optional dependencies: {e}"
        )

    # Semantic Systems (Text)
    try:
        from dam.systems.semantic_systems import handle_semantic_search_query

        world_instance.register_system(handle_semantic_search_query, event_type=SemanticSearchQuery)
        logger.debug("Registered system: handle_semantic_search_query for event SemanticSearchQuery")
    except ImportError as e:
        logger.warning(
            f"Could not import or register handle_semantic_search_query, likely due to missing optional dependencies: {e}"
        )

    # Auto-Tagging System
    try:
        from dam.systems.auto_tagging_system import auto_tag_entities_system

        world_instance.register_system(auto_tag_entities_system, stage=SystemStage.CONTENT_ANALYSIS)
        logger.debug("Registered system: auto_tag_entities_system for stage CONTENT_ANALYSIS")
    except ImportError as e:
        logger.warning(
            f"Could not import or register auto_tag_entities_system, likely due to missing optional dependencies: {e}"
        )

    logger.info(f"Core system registration complete for world: {world_instance.name}")
