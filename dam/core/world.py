import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from sqlalchemy.orm import Session

from dam.core.config import Settings, WorldConfig
from dam.core.config import settings as global_app_settings
from dam.core.database import DatabaseManager
from dam.core.events import BaseEvent
from dam.core.resources import ResourceManager
from dam.core.stages import SystemStage
from dam.core.system_params import WorldContext
from dam.core.systems import WorldScheduler

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
        if not isinstance(world_config, WorldConfig):
            raise TypeError(f"world_config must be an instance of WorldConfig, got {type(world_config)}")

        self.name: str = world_config.name
        self.config: WorldConfig = world_config
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.info(f"Creating minimal World instance: {self.name}")

        self.resource_manager: ResourceManager = ResourceManager()
        self.scheduler: WorldScheduler = WorldScheduler(resource_manager=self.resource_manager)
        self.logger.info(f"Minimal World '{self.name}' instance created. Base resources to be populated externally.")

    def get_db_session(self) -> Session:
        db_mngr = self.get_resource(DatabaseManager)
        return db_mngr.get_db_session()

    def create_db_and_tables(self) -> None:
        self.logger.info(f"Requesting creation of DB tables for World '{self.name}'.")
        db_mngr = self.get_resource(DatabaseManager)
        db_mngr.create_db_and_tables()

    def add_resource(self, instance: Any, resource_type: Optional[Type] = None) -> None:
        self.resource_manager.add_resource(instance, resource_type)
        self.logger.debug(f"Added resource type {resource_type or type(instance)} to World '{self.name}'.")

    def get_resource(self, resource_type: Type[T]) -> T:
        return self.resource_manager.get_resource(resource_type)

    def has_resource(self, resource_type: Type) -> bool:
        return self.resource_manager.has_resource(resource_type)

    def register_system(
        self,
        system_func: Callable[..., Any],
        stage: Optional[SystemStage] = None,
        event_type: Optional[Type[BaseEvent]] = None,
        **kwargs,
    ) -> None:
        if stage and event_type:
            raise ValueError("A system cannot be registered for both a stage and an event type simultaneously.")
        self.scheduler.register_system_for_world(system_func, stage=stage, event_type=event_type, **kwargs)
        if stage:
            self.logger.info(f"System {system_func.__name__} registered for stage {stage.name} in world '{self.name}'.")
        elif event_type:
            self.logger.info(
                f"System {system_func.__name__} registered for event {event_type.__name__} in world '{self.name}'."
            )

    def _get_world_context(self, session: Session) -> WorldContext:
        world_cfg = self.get_resource(WorldConfig)
        return WorldContext(
            session=session,
            world_name=self.name,
            world_config=world_cfg,
        )

    async def execute_stage(self, stage: SystemStage, session: Optional[Session] = None) -> None:
        self.logger.info(f"Executing stage '{stage.name}' for World '{self.name}'.")
        if session:
            world_context = self._get_world_context(session)
            await self.scheduler.execute_stage(stage, world_context)
        else:
            db_session = self.get_db_session()
            try:
                world_context = self._get_world_context(db_session)
                await self.scheduler.execute_stage(stage, world_context)
            finally:
                db_session.close()
                self.logger.debug(f"Session closed after executing stage '{stage.name}' in World '{self.name}'.")

    async def dispatch_event(self, event: BaseEvent, session: Optional[Session] = None) -> None:
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
                self.logger.debug(
                    f"Session closed after dispatching event '{type(event).__name__}' in World '{self.name}'."
                )

    async def execute_one_time_system(
        self, system_func: Callable[..., Any], session: Optional[Session] = None, **kwargs: Any
    ) -> None:
        """
        Executes a single, dynamically provided system function immediately.
        Manages session creation and closure if an external session is not provided.
        """
        self.logger.info(
            f"Executing one-time system '{system_func.__name__}' for World '{self.name}' with kwargs: {kwargs}."
        )
        if session:
            world_context = self._get_world_context(session)
            await self.scheduler.execute_one_time_system(system_func, world_context, **kwargs)
        else:
            db_session = self.get_db_session()
            try:
                world_context = self._get_world_context(db_session)
                await self.scheduler.execute_one_time_system(system_func, world_context, **kwargs)
            finally:
                db_session.close()
                self.logger.debug(
                    f"Session closed after executing one-time system '{system_func.__name__}' in World '{self.name}'."
                )

    def __repr__(self) -> str:
        return f"<World name='{self.name}' config='{self.config!r}'>"


# --- Global World Registry ---
_world_registry: Dict[str, World] = {}


def register_world(world_instance: World) -> None:
    if not isinstance(world_instance, World):
        raise TypeError("Can only register instances of World.")
    if world_instance.name in _world_registry:
        logger.warning(f"World with name '{world_instance.name}' is already registered. Overwriting.")
    _world_registry[world_instance.name] = world_instance
    logger.info(f"World '{world_instance.name}' registered.")


def get_world(world_name: str) -> Optional[World]:
    return _world_registry.get(world_name)


def get_default_world() -> Optional[World]:
    default_name = global_app_settings.DEFAULT_WORLD_NAME
    if default_name:
        return get_world(default_name)
    return None


def get_all_registered_worlds() -> List[World]:
    return list(_world_registry.values())


def unregister_world(world_name: str) -> bool:
    if world_name in _world_registry:
        del _world_registry[world_name]
        logger.info(f"World '{world_name}' unregistered.")
        return True
    logger.warning(f"Attempted to unregister World '{world_name}', but it was not found.")
    return False


def clear_world_registry() -> None:
    count = len(_world_registry)
    _world_registry.clear()
    logger.info(f"Cleared {count} worlds from the registry.")


def create_and_register_world(world_name: str, app_settings: Optional[Settings] = None) -> World:
    current_settings = app_settings or global_app_settings
    logger.info(
        f"Attempting to create and register world: {world_name} using settings: {'provided' if app_settings else 'global'}"
    )
    try:
        world_cfg = current_settings.get_world_config(world_name)
    except ValueError as e:
        logger.error(f"Failed to get configuration for world '{world_name}': {e}")
        raise

    world = World(world_config=world_cfg)

    # Initialize resources and assign to the world instance
    from .world_setup import initialize_world_resources

    populated_resource_manager = initialize_world_resources(world_cfg)
    world.resource_manager = populated_resource_manager

    # Ensure the scheduler uses the populated resource manager
    # If the scheduler was already initialized with the old (empty) one,
    # we need to update its reference or reinitialize it.
    world.scheduler.resource_manager = world.resource_manager
    # Alternatively, if WorldScheduler's __init__ is simple and doesn't do much with RM yet:
    # world.scheduler = WorldScheduler(resource_manager=world.resource_manager)

    world.logger.info(f"World '{world.name}' resources populated and scheduler updated.")

    register_world(world)
    return world


def create_and_register_all_worlds_from_settings(app_settings: Optional[Settings] = None) -> List[World]:
    current_settings = app_settings or global_app_settings
    created_worlds = []
    world_names = current_settings.get_all_world_names()
    logger.info(
        f"Found {len(world_names)} worlds in settings to create and register: {world_names} (using {'provided' if app_settings else 'global'} settings)"
    )
    for name in world_names:
        try:
            world = create_and_register_world(name, app_settings=current_settings)
            created_worlds.append(world)
        except Exception as e:
            logger.error(f"Failed to create or register world '{name}': {e}", exc_info=True)
    return created_worlds
