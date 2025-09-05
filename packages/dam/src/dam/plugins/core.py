import logging

from dam.core.config import settings as global_app_settings
from dam.core.database import DatabaseManager
from dam.core.plugin import Plugin
from dam.core.world import World
from dam.systems.entity_systems import get_or_create_entity_from_stream_handler
from dam.systems.hashing_systems import add_hashes_from_stream_system

logger = logging.getLogger(__name__)


class CorePlugin(Plugin):
    """
    The core plugin for the DAM system.
    """

    def build(self, world: World) -> None:
        """
        Build the core plugin.
        """
        # Logic from initialize_world_resources
        if not world:
            raise ValueError("A valid World instance must be provided.")

        world_config = world.config
        resource_manager = world.resource_manager
        world_name = world_config.name

        world.logger.info(f"Populating base resources for World '{world_name}'...")

        resource_manager.add_resource(world_config, world_config.__class__)
        world.logger.debug(f"Added WorldConfig resource for World '{world_name}'.")

        db_manager = DatabaseManager(
            world_config=world_config,
            testing_mode=global_app_settings.TESTING_MODE,
        )
        resource_manager.add_resource(db_manager, DatabaseManager)
        world.logger.debug(f"Added DatabaseManager resource for World '{world_name}'.")

        world.logger.info(
            f"Base resources populated for World '{world_name}'. Current resources: {list(resource_manager.get_all_resource_types())}"
        )

        # Logic from register_core_systems
        world.register_system(add_hashes_from_stream_system)
        world.register_system(get_or_create_entity_from_stream_handler)
        logger.info(f"Core system registration complete for world: {world.name}")
