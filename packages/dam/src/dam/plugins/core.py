"""Core plugin for the DAM system."""

import logging

from pydantic import Field
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from dam.contexts.providers import MarkedEntityListProvider
from dam.contexts.transaction_manager import TransactionManager
from dam.core.database import DatabaseManager
from dam.core.markers import MarkedEntityList
from dam.core.plugin import Plugin
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.models.config import ConfigComponent, SettingsModel
from dam.models.metadata import ContentMimeTypeComponent
from dam.systems.entity_systems import get_or_create_entity_from_stream_handler
from dam.systems.hashing_systems import add_hashes_from_stream_system
from dam.systems.mime_type_system import (
    check_mime_type_system,
    get_mime_type_system,
    remove_mime_type_system,
    set_mime_type_system,
)
from dam.traits.asset_operation import AssetOperationTrait
from dam.traits.traits import TraitImplementation

logger = logging.getLogger(__name__)


# --- Settings ---


class CoreSettingsModel(SettingsModel):
    """Pydantic model for validating core plugin settings from dam.toml."""

    database_url: str
    alembic_path: str = Field(..., description="Path to the Alembic migrations directory for this world.")


class CoreSettingsComponent(ConfigComponent):
    """ECS component holding the core settings for a world."""

    __tablename__ = "dam_core_config"
    database_url: Mapped[str] = mapped_column(String, nullable=False)
    alembic_path: Mapped[str] = mapped_column(String, nullable=False)


# --- Plugin ---


class CorePlugin(Plugin):
    """The core plugin for the DAM system."""

    Settings = CoreSettingsModel
    SettingsComponent = CoreSettingsComponent

    def build(self, world: World) -> None:
        """Build the core plugin."""
        if not world:
            raise ValueError("A valid World instance must be provided.")

        # Get the validated settings component from world resources
        settings = world.get_resource(CoreSettingsComponent)
        if not settings:
            raise ValueError(
                "CoreSettingsComponent not found in world resources. "
                "The application layer must load and inject it before building the plugin."
            )

        resource_manager = world.resource_manager
        world_name = world.name

        world.logger.info("Populating base resources for World '%s'...", world_name)

        db_manager = DatabaseManager(database_url=settings.database_url)
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

        # Register core traits
        set_mime_type_implementation = TraitImplementation(
            trait=AssetOperationTrait,
            handlers={
                AssetOperationTrait.Add: set_mime_type_system,
                AssetOperationTrait.Check: check_mime_type_system,
                AssetOperationTrait.Remove: remove_mime_type_system,
            },
            name="core.set_mime_type",
            description="Sets the mime type for an asset.",
        )
        world.trait_manager.register(set_mime_type_implementation, ContentMimeTypeComponent)
