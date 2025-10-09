"""Core plugin for the DAM system."""

import logging

from dam.contexts.providers import MarkedEntityListProvider
from dam.contexts.transaction_manager import TransactionManager
from dam.core.database import DatabaseManager
from dam.core.markers import MarkedEntityList
from dam.core.plugin import Plugin
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.systems.entity_systems import get_or_create_entity_from_stream_handler
from dam.systems.hashing_systems import add_hashes_from_stream_system
from dam.systems.mime_type_system import (
    get_mime_type_system,
    set_mime_type_system,
)

logger = logging.getLogger(__name__)


class CorePlugin(Plugin):
    """The core plugin for the DAM system."""

    def build(self, world: World) -> None:
        """Build the core plugin."""
        # Logic from initialize_world_resources
        if not world:
            raise ValueError("A valid World instance must be provided.")

        world_config = world.config
        resource_manager = world.resource_manager
        world_name = world_config.name

        world.logger.info("Populating base resources for World '%s'...", world_name)

        resource_manager.add_resource(world_config, world_config.__class__)
        world.logger.debug("Added WorldConfig resource for World '%s'.", world_name)

        db_manager = DatabaseManager(
            world_config=world_config,
        )
        resource_manager.add_resource(db_manager, DatabaseManager)
        world.logger.debug("Added DatabaseManager resource for World '%s'.", world_name)

        # Register core context providers
        transaction_manager = TransactionManager(db_manager)
        world.register_context_provider(WorldTransaction, transaction_manager)
        world.register_context_provider(MarkedEntityList, MarkedEntityListProvider())
        world.logger.debug("Core context providers registered for World '%s'.", world_name)

        world.logger.info(
            "Base resources populated for World '%s'. Current resources: %s",
            world_name,
            list(resource_manager.get_all_resource_types()),
        )

        # Logic from register_core_systems
        world.register_system(add_hashes_from_stream_system)
        world.register_system(get_or_create_entity_from_stream_handler)
        world.register_system(set_mime_type_system)
        world.register_system(get_mime_type_system)
        logger.info("Core system registration complete for world: %s", world.name)
