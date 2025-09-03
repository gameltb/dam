import asyncio
import inspect
import logging
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    get_args,
    get_origin,
)

from dam.core.commands import BaseCommand, CommandResult, ResultType
from dam.core.events import BaseEvent
from dam.core.resources import ResourceNotFoundError
from dam.core.stages import SystemStage
from dam.core.transaction import EcsTransaction
from dam.models.core.base_component import BaseComponent
from dam.models.core.entity import Entity

if TYPE_CHECKING:
    from dam.core.world import World


SYSTEM_METADATA: Dict[Callable[..., Any], Dict[str, Any]] = {}
logger = logging.getLogger(__name__)

T = TypeVar("T")


def _parse_system_params(func: Callable[..., Any]) -> Dict[str, Any]:
    sig = inspect.signature(func)
    param_info = {}
    for name, param in sig.parameters.items():
        original_param_type = param.annotation
        identity: Optional[str] = None
        actual_type = original_param_type
        marker_component_type: Optional[Type[BaseComponent]] = None
        event_specific_type: Optional[Type[BaseEvent]] = None
        command_specific_type: Optional[Type[BaseCommand[Any]]] = None

        if get_origin(original_param_type) is Annotated:
            annotated_args = get_args(original_param_type)
            actual_type = annotated_args[0]
            string_identities_found = [arg for arg in annotated_args[1:] if isinstance(arg, str)]
            type_based_markers_found = [arg for arg in annotated_args[1:] if inspect.isclass(arg)]

            if len(string_identities_found) > 1:
                logger.warning(
                    f"Parameter '{name}' in system '{func.__name__}' has multiple string annotations: "
                    f"{string_identities_found}. Using the first one: '{string_identities_found[0]}'."
                )
            if string_identities_found:
                identity = string_identities_found[0]

            if identity == "MarkedEntityList":
                marker_component_type = next(
                    (m for m in type_based_markers_found if issubclass(m, BaseComponent)), None
                )
                if not marker_component_type:
                    logger.warning(
                        f"Parameter '{name}' in system '{func.__name__}' is 'MarkedEntityList' but is missing a BaseComponent subclass in Annotated args."
                    )
            elif identity == "Event":
                if inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
                    event_specific_type = actual_type
                else:
                    logger.warning(
                        f"Parameter '{name}' in system '{func.__name__}' is 'Event' but its type '{actual_type}' is not a BaseEvent subclass."
                    )
            elif identity == "Command":
                if inspect.isclass(actual_type) and issubclass(actual_type, BaseCommand):
                    command_specific_type = actual_type
                else:
                    logger.warning(
                        f"Parameter '{name}' in system '{func.__name__}' is 'Command' but its type '{actual_type}' is not a BaseCommand subclass."
                    )

        if not identity:
            if inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
                identity = "Event"
                event_specific_type = actual_type
            elif inspect.isclass(actual_type) and issubclass(actual_type, BaseCommand):
                identity = "Command"
                command_specific_type = actual_type

        if identity == "Event" and not event_specific_type:
            if inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
                event_specific_type = actual_type
            else:
                logger.warning(
                    f"Parameter '{name}' in system '{func.__name__}' resolved to 'Event' identity, but its type '{actual_type}' is not a BaseEvent subclass."
                )
        if identity == "Command" and not command_specific_type:
            if inspect.isclass(actual_type) and issubclass(actual_type, BaseCommand):
                command_specific_type = actual_type
            else:
                logger.warning(
                    f"Parameter '{name}' in system '{func.__name__}' resolved to 'Command' identity, but its type '{actual_type}' is not a BaseCommand subclass."
                )

        param_info[name] = {
            "name": name,
            "type_hint": actual_type,
            "identity": identity,
            "marker_component_type": marker_component_type,
            "event_type_hint": event_specific_type,
            "command_type_hint": command_specific_type,
            "is_annotated": get_origin(original_param_type) is Annotated,
            "original_annotation": original_param_type,
        }
    return param_info


def system(stage: SystemStage, **kwargs):
    def decorator(func: Callable[..., Any]):
        param_info = _parse_system_params(func)
        SYSTEM_METADATA[func] = {
            "params": param_info,
            "is_async": inspect.iscoroutinefunction(func),
            "system_type": "stage_system",
            "stage": stage,
            **kwargs,
        }
        return func

    return decorator


def handles_command(command_type: Type[BaseCommand[Any]], **kwargs):
    if not (inspect.isclass(command_type) and issubclass(command_type, BaseCommand)):
        raise TypeError(f"Invalid command_type '{command_type}'. Must be a class that inherits from BaseCommand.")

    def decorator(func: Callable[..., Any]):
        param_info = _parse_system_params(func)
        SYSTEM_METADATA[func] = {
            "params": param_info,
            "is_async": inspect.iscoroutinefunction(func),
            "system_type": "command_handler",
            "handles_command_type": command_type,
            **kwargs,
        }
        has_command_param = any(p_info.get("command_type_hint") == command_type for p_info in param_info.values())
        if not has_command_param:
            found_by_direct_type = any(
                p_info.get("type_hint") == command_type and p_info.get("identity") == "Command"
                for p_info in param_info.values()
            )
            if not found_by_direct_type:
                logger.warning(
                    f"System {func.__name__} registered for command {command_type.__name__} but does not seem to have a parameter matching this command type."
                )
        return func

    return decorator


def listens_for(event_type: Type[BaseEvent], **kwargs):
    if not (inspect.isclass(event_type) and issubclass(event_type, BaseEvent)):
        raise TypeError(f"Invalid event_type '{event_type}'. Must be a class that inherits from BaseEvent.")

    def decorator(func: Callable[..., Any]):
        param_info = _parse_system_params(func)
        SYSTEM_METADATA[func] = {
            "params": param_info,
            "is_async": inspect.iscoroutinefunction(func),
            "system_type": "event_handler",
            "listens_for_event_type": event_type,
            **kwargs,
        }
        has_event_param = any(p_info.get("event_type_hint") == event_type for p_info in param_info.values())
        if not has_event_param:
            found_by_direct_type = any(
                p_info.get("type_hint") == event_type and p_info.get("identity") == "Event"
                for p_info in param_info.values()
            )
            if not found_by_direct_type:
                logger.warning(
                    f"System {func.__name__} registered for event {event_type.__name__} but does not seem to have a parameter matching this event type."
                )
        return func

    return decorator


class WorldScheduler:
    def __init__(self, world: "World"):
        self.world = world
        self.resource_manager = world.resource_manager
        self.system_registry: Dict[SystemStage, List[Callable[..., Any]]] = defaultdict(list)
        self.event_handler_registry: Dict[Type[BaseEvent], List[Callable[..., Any]]] = defaultdict(list)
        self.command_handler_registry: Dict[Type[BaseCommand[Any]], List[Callable[..., Any]]] = defaultdict(list)
        self.system_metadata = SYSTEM_METADATA
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def register_system_for_world(
        self,
        system_func: Callable[..., Any],
        stage: Optional[SystemStage] = None,
        event_type: Optional[Type[BaseEvent]] = None,
        command_type: Optional[Type[BaseCommand[Any]]] = None,
        **kwargs,
    ):
        if system_func not in self.system_metadata:
            self.logger.warning(
                f"System function {system_func.__name__} is not in global SYSTEM_METADATA. "
                "Attempting to parse its parameters now. Ensure systems are decorated."
            )
            if not _parse_system_params(system_func):
                self.logger.error(
                    f"Failed to parse parameters for undecorated system {system_func.__name__}. Cannot register."
                )
                return
        if stage:
            self.system_registry[stage].append(system_func)
        elif event_type:
            self.event_handler_registry[event_type].append(system_func)
        elif command_type:
            self.command_handler_registry[command_type].append(system_func)
        else:
            self.logger.error(
                f"System {system_func.__name__} must be registered with either a stage, an event_type, or a command_type."
            )

    async def execute_stage(self, stage: SystemStage, transaction: EcsTransaction):
        self.logger.info(f"Executing stage: {stage.name} for world: {self.world.name}")
        systems_to_run = self.system_registry.get(stage, [])
        if not systems_to_run:
            self.logger.info(f"No systems registered for stage {stage.name} in world {self.world.name}")
            return

        for system_func in systems_to_run:
            await self._execute_system_func(system_func, transaction, event_object=None, command_object=None)

    async def dispatch_event(self, event: BaseEvent, transaction: EcsTransaction):
        event_type = type(event)
        self.logger.info(f"Dispatching event: {event_type.__name__} for world: {self.world.name}")
        handlers_to_run = self.event_handler_registry.get(event_type, [])
        if not handlers_to_run:
            self.logger.info(
                f"No event handlers registered for event type {event_type.__name__} in world {self.world.name}"
            )
            return

        for handler_func in handlers_to_run:
            await self._execute_system_func(handler_func, transaction, event_object=event, command_object=None)

    async def dispatch_command(
        self, command: BaseCommand[ResultType], transaction: EcsTransaction
    ) -> CommandResult[ResultType]:
        command_type = type(command)
        self.logger.info(f"Dispatching command: {command_type.__name__} for world: {self.world.name}")
        handlers_to_run = self.command_handler_registry.get(command_type, [])
        if not handlers_to_run:
            self.logger.info(
                f"No command handlers registered for command type {command_type.__name__} in world {self.world.name}"
            )
            return CommandResult(results=[])

        for handler_func in handlers_to_run:
            result = await self._execute_system_func(
                handler_func, transaction, event_object=None, command_object=command
            )
            if result is not None:
                # First handler to return a non-None result wins
                return CommandResult(results=[result])

        # If no handler returned a result, return a CommandResult with an empty list
        return CommandResult(results=[])

    async def run_all_stages(self, transaction: EcsTransaction):
        self.logger.info(f"Attempting to run all stages for world: {self.world.name}")
        ordered_stages = sorted(list(SystemStage), key=lambda s: s.value if isinstance(s.value, int) else str(s.value))
        for stage in ordered_stages:
            self.logger.info(f"Running stage {stage.name} as part of run_all_stages for world {self.world.name}.")
            await self.execute_stage(stage, transaction)
        self.logger.info(f"Finished running all stages for world: {self.world.name}")

    async def _resolve_dependencies(
        self,
        system_func: Callable[..., Any],
        transaction: EcsTransaction,
        event_object: Optional[BaseEvent] = None,
        command_object: Optional[BaseCommand[Any]] = None,
        **additional_kwargs: Any,
    ) -> Dict[str, Any]:
        metadata = self.system_metadata.get(system_func)
        if not metadata:
            self.logger.warning(
                f"No pre-registered metadata for system {system_func.__name__} in world '{self.world.name}'. Attempting dynamic parameter parsing."
            )
            params_info = _parse_system_params(system_func)
            metadata = {"params": params_info, "is_async": inspect.iscoroutinefunction(system_func)}
            self.system_metadata[system_func] = metadata

        kwargs_to_inject: Dict[str, Any] = {}

        for param_name, value in additional_kwargs.items():
            if param_name in metadata["params"]:
                kwargs_to_inject[param_name] = value
            else:
                self.logger.warning(
                    f"Provided kwarg '{param_name}' for system {system_func.__name__} does not match any system parameter. It will be ignored."
                )

        for param_name, param_meta in metadata["params"].items():
            if param_name in kwargs_to_inject:
                continue

            identity = param_meta["identity"]
            param_type_hint = param_meta["type_hint"]

            if param_type_hint is EcsTransaction:
                kwargs_to_inject[param_name] = transaction
            elif identity == "WorldSession":
                self.logger.warning(
                    f"System '{system_func.__name__}' is injecting WorldSession directly. "
                    "This is deprecated. Please inject EcsTransaction instead."
                )
                kwargs_to_inject[param_name] = transaction.session
            elif identity == "Resource":
                kwargs_to_inject[param_name] = self.resource_manager.get_resource(param_type_hint)
            elif identity == "MarkedEntityList":
                marker_type = param_meta["marker_component_type"]
                if not marker_type or not issubclass(marker_type, BaseComponent):
                    msg = f"System {system_func.__name__} has MarkedEntityList parameter '{param_name}' with invalid or missing marker component type in world '{self.world.name}'."
                    self.logger.error(msg)
                    raise ValueError(msg)
                from sqlalchemy import exists as sql_exists
                from sqlalchemy import select as sql_select

                stmt = sql_select(Entity).where(sql_exists().where(marker_type.entity_id == Entity.id))
                result = await transaction.session.execute(stmt)
                entities_to_process = result.scalars().all()
                kwargs_to_inject[param_name] = entities_to_process
            elif identity == "Event":
                expected_event_type = param_meta["event_type_hint"]
                if event_object and isinstance(event_object, expected_event_type):
                    kwargs_to_inject[param_name] = event_object
                elif expected_event_type is not None and not event_object:
                    msg = f"System {system_func.__name__} parameter '{param_name}' in world '{self.world.name}' expects an event of type {expected_event_type.__name__} but none was provided for injection."
                    self.logger.error(msg)
                    raise ValueError(msg)
            elif identity == "Command":
                expected_command_type = param_meta["command_type_hint"]
                if command_object and isinstance(command_object, expected_command_type):
                    kwargs_to_inject[param_name] = command_object
                elif expected_command_type is not None and not command_object:
                    msg = f"System {system_func.__name__} parameter '{param_name}' in world '{self.world.name}' expects a command of type {expected_command_type.__name__} but none was provided for injection."
                    self.logger.error(msg)
                    raise ValueError(msg)
            else:
                if not (
                    param_type_hint is str
                    or param_type_hint is int
                    or param_type_hint is bool
                    or param_type_hint is float
                    or param_type_hint is list
                    or param_type_hint is dict
                    or param_type_hint is tuple
                    or param_type_hint is set
                    or param_type_hint is type(None)
                ):
                    try:
                        kwargs_to_inject[param_name] = self.resource_manager.get_resource(param_type_hint)
                    except ResourceNotFoundError:
                        self.logger.debug(
                            f"Resource not found for param '{param_name}' (type {param_type_hint}) via direct type injection for {system_func.__name__}. "
                            "It might be provided by additional_kwargs or be optional."
                        )
        return kwargs_to_inject

    async def _execute_system_func(
        self,
        system_func: Callable[..., Any],
        transaction: EcsTransaction,
        event_object: Optional[BaseEvent] = None,
        command_object: Optional[BaseCommand[Any]] = None,
        **additional_kwargs: Any,
    ) -> Any:
        metadata = self.system_metadata.get(system_func)
        is_async_func = inspect.iscoroutinefunction(system_func)
        if metadata:
            is_async_func = metadata["is_async"]

        try:
            kwargs_to_inject = await self._resolve_dependencies(
                system_func, transaction, event_object, command_object, **additional_kwargs
            )
        except Exception as e:
            self.logger.error(
                f"Error resolving dependencies for system {system_func.__name__} in world '{self.world.name}': {e}",
                exc_info=True,
            )
            raise

        self.logger.debug(
            f"Executing system: {system_func.__name__} in world '{self.world.name}' with args: {list(kwargs_to_inject.keys())}"
        )

        result: Any = None
        if is_async_func:
            result = await system_func(**kwargs_to_inject)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: system_func(**kwargs_to_inject))

        if metadata and metadata.get("system_type") == "stage_system":
            for param_name, param_meta in metadata["params"].items():
                if param_meta.get("identity") == "MarkedEntityList" and metadata.get("auto_remove_marker", True):
                    marker_type_to_remove = param_meta["marker_component_type"]
                    entities_processed = kwargs_to_inject.get(param_name, [])
                    if entities_processed and marker_type_to_remove:
                        entity_ids_processed = [entity.id for entity in entities_processed]
                        if entity_ids_processed:
                            from sqlalchemy import delete as sql_delete

                            self.logger.debug(
                                f"Scheduler for world '{self.world.name}': Bulk removing {marker_type_to_remove.__name__} from "
                                f"{len(entity_ids_processed)} entities after system {system_func.__name__}."
                            )
                            stmt = sql_delete(marker_type_to_remove).where(
                                marker_type_to_remove.entity_id.in_(entity_ids_processed)
                            )
                            await transaction.session.execute(stmt)
                            if metadata.get("system_type") == "stage_system":
                                await transaction.session.flush()
        return result

    async def execute_one_time_system(
        self, system_func: Callable[..., Any], transaction: EcsTransaction, **kwargs: Any
    ) -> Any:
        self.logger.info(
            f"Executing one-time system: {system_func.__name__} in world '{self.world.name}' with provided kwargs: {kwargs}"
        )
        try:
            result = await self._execute_system_func(
                system_func, transaction, event_object=None, command_object=None, **kwargs
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Error during execution of one-time system {system_func.__name__} in world '{self.world.name}': {e}. "
                "Session rollback should be handled by the caller.",
                exc_info=True,
            )
            raise
