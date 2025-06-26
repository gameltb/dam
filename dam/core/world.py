import logging
from typing import Optional, Type, Any, List, TypeVar, Dict, Callable # Added Dict and Callable

from sqlalchemy.orm import Session

from dam.core.config import WorldConfig, settings as global_app_settings
from dam.core.database import DatabaseManager
from dam.core.resources import ResourceManager, FileOperationsResource # Assuming FileOperationsResource is standard
from dam.core.systems import WorldScheduler, SYSTEM_METADATA # SYSTEM_REGISTRY, EVENT_HANDLER_REGISTRY removed
from dam.core.system_params import WorldContext
from dam.core.events import BaseEvent
from dam.core.stages import SystemStage
from dam.core.config import WorldConfig # Already imported, ensure it's usable for resource typing

# FileStorageService class does not exist in dam.services.file_storage, it's a module of functions.
# from dam.services.file_storage import FileStorageService # This line causes ImportError

logger = logging.getLogger(__name__)

# Type variable for generic resource types
T = TypeVar("T")

class World:
    """
    Represents an isolated ECS (Entity Component System) world.

    Each World encapsulates its own configuration, database connection,
    resource manager, and system scheduler. This allows for multiple
    independent worlds to coexist within the same application, each with
    potentially different settings, data stores, and behaviors.
    """

    def __init__(self, world_config: WorldConfig):
        """
        Initializes a new World instance.

        Args:
            world_config: The configuration object for this specific world.
        """
        if not isinstance(world_config, WorldConfig):
            raise TypeError(f"world_config must be an instance of WorldConfig, got {type(world_config)}")

        self.name: str = world_config.name
        self.config: WorldConfig = world_config # Keep a direct reference to its own config
        self.logger = logging.getLogger(f"{__name__}.{self.name}") # World-specific logger
        self.logger.info(f"Creating minimal World instance: {self.name}")

        self.resource_manager: ResourceManager = ResourceManager()
        self.scheduler: WorldScheduler = WorldScheduler(resource_manager=self.resource_manager)

        # Base resources (DatabaseManager, FileStorageService, WorldConfig as resource, FileOperationsResource)
        # will be added by an external setup function like `populate_base_resources(world)`.

        self.logger.info(f"Minimal World '{self.name}' instance created. Base resources to be populated externally.")

    # Removed _initialize_database_manager and _initialize_additional_default_resources
    # Their logic will be in the new world_setup.py:populate_base_resources


    # --- Database Access (now typically via injected DatabaseManager or direct Session from World) ---
    def get_db_session(self) -> Session:
        """
        Provides a new database session for this World.
        The caller is responsible for closing the session.
        This method relies on the DatabaseManager resource.
        """
        db_mngr = self.get_resource(DatabaseManager)
        return db_mngr.get_db_session()

    def create_db_and_tables(self) -> None:
        """
        Creates all database tables for this World.
        Delegates to this World's DatabaseManager resource.
        """
        self.logger.info(f"Requesting creation of DB tables for World '{self.name}'.")
        db_mngr = self.get_resource(DatabaseManager)
        db_mngr.create_db_and_tables()

    # --- Resource Management ---
    def add_resource(self, instance: Any, resource_type: Optional[Type] = None) -> None:
        """Adds a resource instance to this World's ResourceManager."""
        self.resource_manager.add_resource(instance, resource_type)
        self.logger.debug(f"Added resource type {resource_type or type(instance)} to World '{self.name}'.")

    def get_resource(self, resource_type: Type[T]) -> T:
        """Retrieves a resource instance by its type from this World's ResourceManager."""
        return self.resource_manager.get_resource(resource_type)

    def has_resource(self, resource_type: Type) -> bool:
        """Checks if a resource of the given type is registered in this World."""
        return self.resource_manager.has_resource(resource_type)

    # --- System Registration (NEW) ---
    def register_system(self, system_func: Callable[..., Any], stage: Optional[SystemStage] = None, event_type: Optional[Type[BaseEvent]] = None, **kwargs) -> None:
        """
        Registers a system function to this specific World's scheduler.
        The system will be associated with either a stage or an event type.
        This replaces global registration for systems intended to run in this world.
        """
        if stage and event_type:
            raise ValueError("A system cannot be registered for both a stage and an event type simultaneously.")

        # The WorldScheduler will now need its own registries, or we adapt the global ones
        # For now, let's assume WorldScheduler is modified to have instance-level registries.
        # Or, the World object itself manages a list of systems and passes them to scheduler.
        # Let's refine this: WorldScheduler can still use global metadata (SYSTEM_METADATA)
        # but its internal lists of systems to run per stage/event should be specific to the World instance.

        # This requires modification to WorldScheduler to accept system registrations
        # or for World to maintain its own list of systems and pass them to the scheduler.
        # For now, let's assume WorldScheduler is adapted.
        self.scheduler.register_system_for_world(system_func, stage=stage, event_type=event_type, **kwargs)
        if stage:
            self.logger.info(f"System {system_func.__name__} registered for stage {stage.name} in world '{self.name}'.")
        elif event_type:
            self.logger.info(f"System {system_func.__name__} registered for event {event_type.__name__} in world '{self.name}'.")


    # --- System Execution & Event Dispatch ---
    def _get_world_context(self, session: Session) -> WorldContext:
        """Helper to create a WorldContext for system execution."""
        # WorldConfig is now available as a resource, so systems can inject it.
        # WorldContext might not need to carry it directly if systems can get it from resource_manager.
        # However, keeping it for now for compatibility or if direct access is faster.
        world_cfg = self.get_resource(WorldConfig) # Get it from the resource manager
        return WorldContext(
            session=session,
            world_name=self.name,
            world_config=world_cfg, # Pass the specific config
        )

    async def execute_stage(self, stage: SystemStage, session: Optional[Session] = None) -> None:
        """
        Executes all systems registered for a specific SystemStage within this World.
        If a session is not provided, a new one will be created for the scope of this stage.
        The provided or created session will be committed or rolled back by the scheduler.

        Args:
            stage: The SystemStage to execute.
            session: (Optional) An existing SQLAlchemy session to use. If None, a new
                     session is created and managed by this method for the stage.
        """
        self.logger.info(f"Executing stage '{stage.name}' for World '{self.name}'.")
        if session:
            world_context = self._get_world_context(session)
            await self.scheduler.execute_stage(stage, world_context)
        else:
            # Create a session for this stage execution
            # The scheduler's execute_stage will handle commit/rollback for this session.
            db_session = self.get_db_session()
            try:
                world_context = self._get_world_context(db_session)
                await self.scheduler.execute_stage(stage, world_context)
            finally:
                # If the scheduler doesn't close the session it created internally,
                # we should close it here.
                # Current WorldScheduler's execute_stage commits/rolls back but doesn't close.
                # For sessions created and passed by the World, it's cleaner if the World manages its closure.
                # However, if execute_stage handles commit/rollback, the session is likely "done".
                # Let's assume the session used by execute_stage should be closed after.
                db_session.close()
                self.logger.debug(f"Session closed after executing stage '{stage.name}' in World '{self.name}'.")


    async def dispatch_event(self, event: BaseEvent, session: Optional[Session] = None) -> None:
        """
        Dispatches an event to all relevant event handlers within this World.
        If a session is not provided, a new one will be created for the scope of this event dispatch.
        The provided or created session will be committed or rolled back by the scheduler.

        Args:
            event: The event instance to dispatch.
            session: (Optional) An existing SQLAlchemy session to use.
        """
        self.logger.info(f"Dispatching event '{type(event).__name__}' for World '{self.name}'.")
        if session:
            world_context = self._get_world_context(session)
            await self.scheduler.dispatch_event(event, world_context)
        else:
            db_session = self.get_db_session()
            try:
                world_context = self._get_world_context(db_session)
                await self.scheduler.dispatch_event(event, world_context)
            finally:
                db_session.close()
                self.logger.debug(f"Session closed after dispatching event '{type(event).__name__}' in World '{self.name}'.")

    def __repr__(self) -> str:
        return f"<World name='{self.name}' config='{self.config!r}'>"


# --- Global World Registry (Optional but helpful for managing multiple worlds) ---
# This is a simple dictionary-based registry. More sophisticated management might be needed.
_world_registry: Dict[str, World] = {}

def register_world(world_instance: World) -> None:
    """Registers a World instance in the global registry."""
    if not isinstance(world_instance, World):
        raise TypeError("Can only register instances of World.")
    if world_instance.name in _world_registry:
        logger.warning(f"World with name '{world_instance.name}' is already registered. Overwriting.")
    _world_registry[world_instance.name] = world_instance
    logger.info(f"World '{world_instance.name}' registered.")

def get_world(world_name: str) -> Optional[World]:
    """Retrieves a World instance from the registry by its name."""
    return _world_registry.get(world_name)

def get_default_world() -> Optional[World]:
    """Retrieves the default World instance from the registry."""
    default_name = global_app_settings.DEFAULT_WORLD_NAME
    if default_name:
        return get_world(default_name)
    return None

def get_all_registered_worlds() -> List[World]:
    """Returns a list of all registered World instances."""
    return list(_world_registry.values())

def unregister_world(world_name: str) -> bool:
    """Removes a World instance from the registry. Returns True if found and removed."""
    if world_name in _world_registry:
        del _world_registry[world_name]
        logger.info(f"World '{world_name}' unregistered.")
        return True
    logger.warning(f"Attempted to unregister World '{world_name}', but it was not found.")
    return False

def clear_world_registry() -> None:
    """Clears all World instances from the registry."""
    count = len(_world_registry)
    _world_registry.clear()
    logger.info(f"Cleared {count} worlds from the registry.")


from dam.core.config import Settings # Import Settings for type hinting

# ... (other code) ...

def create_and_register_world(world_name: str, app_settings: Optional[Settings] = None) -> World:
    """
    Factory function to create a World instance based on provided or global settings,
    initialize it, and register it.

    Args:
        world_name: The name of the world to create, must exist in global_app_settings.

    Returns:
        The created and registered World instance.

    Raises:
        ValueError: If the world_name is not found in the global configurations.
    """
    current_settings = app_settings or global_app_settings
    logger.info(f"Attempting to create and register world: {world_name} using settings: {'provided' if app_settings else 'global'}")
    try:
        world_cfg = current_settings.get_world_config(world_name)
    except ValueError as e:
        logger.error(f"Failed to get configuration for world '{world_name}': {e}")
        raise

    world = World(world_config=world_cfg)

    # Populate base resources using the new setup function
    from .world_setup import populate_base_resources # Local import if preferred
    populate_base_resources(world)

    # Optional: Create DB and tables upon world creation/registration if desired
    # world.create_db_and_tables() # Be cautious with this in production environments

    register_world(world)
    return world

def create_and_register_all_worlds_from_settings(app_settings: Optional[Settings] = None) -> List[World]:
    """
    Creates and registers World instances for all world configurations found in the
    provided or global app_settings.
    """
    current_settings = app_settings or global_app_settings
    created_worlds = []
    world_names = current_settings.get_all_world_names()
    logger.info(f"Found {len(world_names)} worlds in settings to create and register: {world_names} (using {'provided' if app_settings else 'global'} settings)")
    for name in world_names:
        try:
            # Pass the current_settings instance down
            world = create_and_register_world(name, app_settings=current_settings)
            created_worlds.append(world)
        except Exception as e:
            logger.error(f"Failed to create or register world '{name}': {e}", exc_info=True)
            # Decide if one failure should stop all, or just skip this world
    return created_worlds


# Example of how worlds might be initialized at application startup:
# if __name__ == "__main__":
#     # Configure logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
#     # Load settings (this happens when config.py is imported, settings is global)
#     logger.info(f"Application settings loaded. Default world: {global_app_settings.DEFAULT_WORLD_NAME}")
#     logger.info(f"Available world configurations: {global_app_settings.get_all_world_names()}")
#
#     # Create and register all worlds defined in settings
#     initialized_worlds = create_and_register_all_worlds_from_settings()
#     logger.info(f"Successfully initialized and registered {len(initialized_worlds)} worlds.")
#
#     # Example: Get the default world and operate on it
#     default_w = get_default_world()
#     if default_w:
#         logger.info(f"Operating on default world: {default_w.name}")
#         # default_w.create_db_and_tables() # If not done during creation
#         # session = default_w.get_db_session()
#         # try:
#         #     # ... use session ...
#         # finally:
#         #     session.close()
#
#         # Example: Get a specific world
#         # test_world = get_world("testing_world") # Assuming "testing_world" is configured
#         # if test_world:
#         #    logger.info(f"Found testing world: {test_world.name}")
#
#     else:
#         logger.error("Could not retrieve the default world.")
