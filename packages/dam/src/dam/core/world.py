import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Type, TypeVar, cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dam.core.commands import BaseCommand, ResultType
from dam.core.config import Settings, WorldConfig
from dam.core.config import settings as global_app_settings
from dam.core.database import DatabaseManager
from dam.core.enums import ExecutionStrategy
from dam.core.events import BaseEvent
from dam.core.executor import SystemExecutor
from dam.core.plugin import Plugin
from dam.core.resources import ResourceManager
from dam.core.stages import SystemStage
from dam.core.systems import WorldScheduler
from dam.core.transaction import EcsTransaction, active_transaction
from dam.system_events import BaseSystemEvent, SystemResultEvent

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
        self.scheduler: WorldScheduler = WorldScheduler(world=self)
        self._registered_plugin_types: set[Type[Plugin]] = set()
        self.add_resource(self)
        self.logger.info(f"Minimal World '{self.name}' instance created. Base resources to be populated externally.")

    @property
    def db_session_maker(self) -> async_sessionmaker[AsyncSession]:
        """Returns the SQLAlchemy sessionmaker for this world's database."""
        db_mngr = self.get_resource(DatabaseManager)
        return db_mngr.session_local  # Expose the sessionmaker instance

    def get_db_session(self) -> AsyncSession:  # Changed to AsyncSession
        db_mngr = self.get_resource(DatabaseManager)
        # get_db_session in DBManager returns result of session_local(), which is AsyncSession for async engine
        return db_mngr.get_db_session()

    async def create_db_and_tables(self) -> None:
        self.logger.info(f"Requesting creation of DB tables for World '{self.name}'.")
        db_mngr = self.get_resource(DatabaseManager)
        await db_mngr.create_db_and_tables()

    def add_resource(self, instance: Any, resource_type: Optional[Type[Any]] = None) -> None:
        self.resource_manager.add_resource(instance, resource_type)
        self.logger.debug(f"Added resource type {resource_type or type(instance)} to World '{self.name}'.")

    def get_resource(self, resource_type: Type[T]) -> T:
        return self.resource_manager.get_resource(resource_type)

    def has_resource(self, resource_type: Type[Any]) -> bool:
        return self.resource_manager.has_resource(resource_type)

    def add_plugin(self, plugin: Plugin) -> "World":
        plugin_type = type(plugin)
        if plugin_type not in self._registered_plugin_types:
            self.logger.info(f"Adding plugin {plugin_type.__name__} to world '{self.name}'.")
            plugin.build(self)
            self._registered_plugin_types.add(plugin_type)
        else:
            self.logger.debug(f"Plugin {plugin_type.__name__} is already registered in world '{self.name}'. Skipping.")
        return self

    def register_system(
        self,
        system_func: Callable[..., Any],
        stage: Optional[SystemStage] = None,
        event_type: Optional[Type[BaseEvent]] = None,
        command_type: Optional[Type[BaseCommand[Any]]] = None,
        **kwargs: Any,
    ) -> None:
        num_triggers = sum(1 for trigger in [stage, event_type, command_type] if trigger is not None)
        if num_triggers > 1:
            raise ValueError("A system can only be registered for one trigger type (stage, event, or command).")

        self.scheduler.register_system_for_world(
            system_func, stage=stage, event_type=event_type, command_type=command_type, **kwargs
        )

        if stage:
            self.logger.info(f"System {system_func.__name__} registered for stage {stage.name} in world '{self.name}'.")
        elif event_type:
            self.logger.info(
                f"System {system_func.__name__} registered for event {event_type.__name__} in world '{self.name}'."
            )
        elif command_type:
            self.logger.info(
                f"System {system_func.__name__} registered for command {command_type.__name__} in world '{self.name}'."
            )

    async def execute_stage(self, stage: SystemStage) -> None:
        self.logger.info(f"Executing stage '{stage.name}' for World '{self.name}'.")

        transaction = active_transaction.get()
        if transaction:
            self.logger.debug(f"Joining existing transaction for stage '{stage.name}'.")
            await self.scheduler.execute_stage(stage, transaction)
        else:
            self.logger.debug(f"Creating new transaction for top-level stage '{stage.name}'.")
            db_session = self.get_db_session()
            new_transaction = EcsTransaction(db_session)
            token = active_transaction.set(new_transaction)
            try:
                await self.scheduler.execute_stage(stage, new_transaction)
                await db_session.commit()
            except Exception as e:
                self.logger.exception(f"Exception in top-level stage '{stage.name}', rolling back.")
                await db_session.rollback()
                from dam.core.exceptions import StageExecutionError

                raise StageExecutionError(
                    message=f"Top-level stage '{stage.name}' failed.",
                    stage_name=stage.name,
                    original_exception=e,
                ) from e
            finally:
                await db_session.close()
                active_transaction.reset(token)
                self.logger.debug(f"Transaction closed for stage '{stage.name}'.")

    async def dispatch_event(self, event: BaseEvent) -> None:
        self.logger.info(f"Dispatching event '{type(event).__name__}' for World '{self.name}'.")

        transaction = active_transaction.get()
        if transaction:
            self.logger.debug(f"Joining existing transaction for event '{type(event).__name__}'.")
            await self.scheduler.dispatch_event(event, transaction)
        else:
            self.logger.debug(f"Creating new transaction for top-level event '{type(event).__name__}'.")
            db_session = self.get_db_session()
            new_transaction = EcsTransaction(db_session)
            token = active_transaction.set(new_transaction)
            try:
                await self.scheduler.dispatch_event(event, new_transaction)
                await db_session.commit()
            except Exception as e:
                self.logger.exception(f"Exception in top-level event '{type(event).__name__}', rolling back.")
                await db_session.rollback()
                from dam.core.exceptions import EventHandlingError

                raise EventHandlingError(
                    message=f"Top-level event '{type(event).__name__}' failed.",
                    event_type=type(event).__name__,
                    original_exception=e,
                ) from e
            finally:
                await db_session.close()
                active_transaction.reset(token)
                self.logger.debug(f"Transaction closed for event '{type(event).__name__}'.")

    def dispatch_command(self, command: BaseCommand[ResultType]) -> SystemExecutor[ResultType]:
        self.logger.info(f"Dispatching command '{type(command).__name__}' for World '{self.name}'.")

        async def _transaction_wrapper() -> AsyncGenerator[BaseSystemEvent, None]:
            """
            A generator that wraps the command execution within a transaction,
            ensuring commit/rollback logic is correctly applied after the
            command's event stream is fully consumed.
            """
            transaction = active_transaction.get()
            if transaction:
                # Already in a transaction, so just execute and yield.
                executor = self.scheduler.dispatch_command(command, transaction)
                async for event in executor:
                    yield event
                return

            # Not in a transaction, so create a new one.
            self.logger.debug(f"Creating new transaction for top-level command '{type(command).__name__}'.")
            db_session = self.get_db_session()
            new_transaction = EcsTransaction(db_session)
            token = active_transaction.set(new_transaction)
            try:
                executor = self.scheduler.dispatch_command(command, new_transaction)
                async for event in executor:
                    yield event
                await db_session.commit()
            except Exception:
                self.logger.exception(f"Exception in top-level command '{type(command).__name__}', rolling back.")
                await db_session.rollback()
                raise  # Re-raise the original exception
            finally:
                await db_session.close()
                active_transaction.reset(token)
                self.logger.debug(f"Transaction closed for command '{type(command).__name__}'.")

        # The returned executor runs the wrapper, which in turn runs the actual command executor.
        # This ensures the lazy execution happens inside the transaction.
        return SystemExecutor([_transaction_wrapper()], ExecutionStrategy.SERIAL)

    async def execute_one_time_system(self, system_func: Callable[..., Any], **kwargs: Any) -> Any:
        """
        Executes a single, dynamically provided system function immediately.
        Manages session creation and closure if an external session is not provided.
        Returns the result of the system function.
        """
        self.logger.info(
            f"Executing one-time system '{system_func.__name__}' for World '{self.name}' with kwargs: {kwargs}."
        )

        async def _run_and_get_result(transaction: EcsTransaction) -> Any:
            executor = self.scheduler.execute_one_time_system(system_func, transaction, **kwargs)
            event: BaseSystemEvent
            async for event in executor:
                if isinstance(event, SystemResultEvent):
                    typed_event = cast(SystemResultEvent[Any], event)
                    return typed_event.result
            return None

        transaction = active_transaction.get()
        if transaction:
            self.logger.debug(f"Joining existing transaction for one-time system '{system_func.__name__}'.")
            return await _run_and_get_result(transaction)
        else:
            self.logger.debug(f"Creating new transaction for top-level one-time system '{system_func.__name__}'.")
            db_session = self.get_db_session()
            new_transaction = EcsTransaction(db_session)
            token = active_transaction.set(new_transaction)
            try:
                result = await _run_and_get_result(new_transaction)
                await new_transaction.session.commit()
                return result
            except Exception:
                self.logger.exception(f"Exception in top-level one-time system '{system_func.__name__}', rolling back.")
                await new_transaction.session.rollback()
                raise
            finally:
                await new_transaction.session.close()
                active_transaction.reset(token)
                self.logger.debug(f"Transaction closed for one-time system '{system_func.__name__}'.")

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[EcsTransaction, None]:
        """Provides a transactional context for database operations."""
        transaction = active_transaction.get()
        if transaction:
            self.logger.debug("Joining existing transaction.")
            yield transaction
            return

        self.logger.debug("Creating new top-level transaction.")
        db_session = self.get_db_session()
        new_transaction = EcsTransaction(db_session)
        token = active_transaction.set(new_transaction)
        try:
            yield new_transaction
            await db_session.commit()
        except Exception:
            self.logger.exception("Exception in top-level transaction, rolling back.")
            await db_session.rollback()
            raise
        finally:
            await db_session.close()
            active_transaction.reset(token)
            self.logger.debug("Top-level transaction closed.")

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

    from dam.plugins.core import CorePlugin

    world.add_plugin(CorePlugin())

    # The scheduler was initialized with world.resource_manager. Since initialize_world_resources
    # modifies that same instance, the scheduler should already have the correct reference.
    # Explicitly setting it again like world.scheduler.resource_manager = world.resource_manager is harmless
    # but usually not necessary if the instance itself was modified.
    # For clarity and safety, ensuring the scheduler sees the potentially *reconfigured* manager is good.
    world.scheduler.resource_manager = world.resource_manager

    world.logger.info(f"World '{world.name}' resources populated and scheduler updated.")

    register_world(world)
    return world


def create_and_register_all_worlds_from_settings(app_settings: Optional[Settings] = None) -> List[World]:
    current_settings = app_settings or global_app_settings
    created_worlds: List[World] = []
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
