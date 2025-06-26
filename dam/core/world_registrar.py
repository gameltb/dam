import logging
from typing import TYPE_CHECKING

from dam.core.events import (
    AssetFileIngestionRequested,
    AssetReferenceIngestionRequested,
    FindEntityByHashQuery,
    FindSimilarImagesQuery,
    # Add other event types used by core systems if any
)
from dam.core.stages import SystemStage
from dam.systems.asset_lifecycle_systems import (
    handle_asset_file_ingestion_request,
    handle_asset_reference_ingestion_request,
    handle_find_entity_by_hash_query,
    handle_find_similar_images_query,
    # Import other asset lifecycle systems if they are core
)
from dam.systems.metadata_systems import (
    extract_metadata_on_asset_ingested,
    # Import other metadata systems if they are core
)

if TYPE_CHECKING:
    from dam.core.world import World

logger = logging.getLogger(__name__)


def register_core_systems(world_instance: "World") -> None:
    """
    Registers all standard, core systems for a given world instance.
    This ensures consistency in system registration across different application entry points.
    """
    logger.info(f"Registering core systems for world: {world_instance.name}")

    # Metadata Systems
    world_instance.register_system(extract_metadata_on_asset_ingested, stage=SystemStage.METADATA_EXTRACTION)
    logger.debug("Registered system: extract_metadata_on_asset_ingested for stage METADATA_EXTRACTION")

    # Asset Lifecycle Systems (Event-based)
    world_instance.register_system(
        handle_asset_file_ingestion_request,  # Standardized name
        event_type=AssetFileIngestionRequested,
    )
    logger.debug("Registered system: handle_asset_file_ingestion_request for event AssetFileIngestionRequested")

    world_instance.register_system(
        handle_asset_reference_ingestion_request,  # Standardized name
        event_type=AssetReferenceIngestionRequested,
    )
    logger.debug(
        "Registered system: handle_asset_reference_ingestion_request for event AssetReferenceIngestionRequested"
    )

    world_instance.register_system(handle_find_entity_by_hash_query, event_type=FindEntityByHashQuery)
    logger.debug("Registered system: handle_find_entity_by_hash_query for event FindEntityByHashQuery")

    world_instance.register_system(handle_find_similar_images_query, event_type=FindSimilarImagesQuery)
    logger.debug("Registered system: handle_find_similar_images_query for event FindSimilarImagesQuery")

    # Add registrations for any other core systems here...
    # For example:
    # world_instance.register_system(some_other_core_system, stage=SystemStage.SOME_STAGE)
    # logger.debug(f"Registered system: some_other_core_system for stage SOME_STAGE")

    logger.info(f"Core system registration complete for world: {world_instance.name}")


# To make this auto-discovery friendly in the future (as discussed in review):
# One could extend this by having systems self-declare their registration info,
# and this function could discover them from designated modules.
# For now, explicit registration here maintains clarity.
