# pyright: basic
import asyncio
import inspect
import logging
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    get_args,
    get_origin,
)

from dam.core.commands import BaseCommand, ResultType
from dam.core.enums import ExecutionStrategy
from dam.core.events import BaseEvent
from dam.core.executor import SystemExecutor
from dam.core.resources import ResourceNotFoundError
from dam.core.stages import SystemStage
from dam.system_events import BaseSystemEvent, SystemResultEvent
from dam.core.transaction import EcsTransaction
from dam.models.core.base_component import BaseComponent
from dam.models.core.entity import Entity

if TYPE_CHECKING:
    from dam.core.world import World


SYSTEM_METADATA: Dict[Callable[..., Any], Dict[str, Any]] = {}
logger = logging.getLogger(__name__)

T = TypeVar("T")


def _parse_system_params(func: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(func)
    param_info: dict[str, Any] = {}
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


def system(
    func: Optional[Callable[..., Any]] = None,
    *,
    on_stage: Optional[SystemStage] = None,
    on_command: Optional[Type[BaseCommand[Any]]] = None,
    on_event: Optional[Type[BaseEvent]] = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        param_info = _parse_system_params(f)

        system_type = "vanilla"
        if on_stage:
            system_type = "stage_system"
        elif on_command:
            system_type = "command_handler"
        elif on_event:
            system_type = "event_handler"

        SYSTEM_METADATA[f] = {
            "params": param_info,
            "is_async": inspect.iscoroutinefunction(f),
            "system_type": system_type,
            "stage": on_stage,
            "handles_command_type": on_command,
            "listens_for_event_type": on_event,
            **kwargs,
        }
        return f

    if func:
        return decorator(func)
    return decorator


class WorldScheduler:
    def __init__(self, world: "World") -> None:
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
        **kwargs: Any,
    ) -> None:
        metadata = self.system_metadata.get(system_func)
        if not metadata:
            self.logger.warning(
                f"System function {system_func.__name__} is not decorated with @system. "
                "Registration relies on explicit parameters."
            )
            # Fallback to explicit parameters for undecorated systems
            if stage:
                self.system_registry[stage].append(system_func)
            elif event_type:
                self.event_handler_registry[event_type].append(system_func)
            elif command_type:
                self.command_handler_registry[command_type].append(system_func)
            else:
                self.logger.error(
                    f"Undecorated system {system_func.__name__} must be registered with an explicit stage, event_type, or command_type."
                )
            return

        # Use metadata from the decorator
        reg_stage = metadata.get("stage")
        reg_command = metadata.get("handles_command_type")
        reg_event = metadata.get("listens_for_event_type")

        if reg_stage:
            self.system_registry[reg_stage].append(system_func)
            self.logger.info(f"Registered system '{system_func.__name__}' for stage '{reg_stage.name}'.")
        elif reg_command:
            # Check for return type mismatch
            command_result_type = None
            for base in getattr(reg_command, "__orig_bases__", []):
                if get_origin(base) is BaseCommand:
                    args = get_args(base)
                    if args:
                        command_result_type = args[0]
                        break

            sig = inspect.signature(system_func)
            handler_return_type = sig.return_annotation

            if handler_return_type is inspect.Signature.empty:
                handler_return_type = Any

            if command_result_type is not None and command_result_type is not Any:
                # Normalize types by getting their origin if they are generic aliases
                # This handles cases like list vs. List, tuple vs. Tuple
                normalized_command_type = get_origin(command_result_type) or command_result_type
                normalized_handler_type = get_origin(handler_return_type) or handler_return_type

                # Normalize None to type(None) for comparison
                if normalized_handler_type is None:
                    normalized_handler_type = type(None)

                if normalized_command_type != normalized_handler_type:
                    self.logger.warning(
                        f"Potential return type mismatch for command '{reg_command.__name__}'. "
                        f"Handler '{system_func.__name__}' is annotated to return '{handler_return_type}' "
                        f"but command expects '{command_result_type}'."
                    )

            self.command_handler_registry[reg_command].append(system_func)
            self.logger.info(f"Registered system '{system_func.__name__}' for command '{reg_command.__name__}'.")
        elif reg_event:
            self.event_handler_registry[reg_event].append(system_func)
            self.logger.info(f"Registered system '{system_func.__name__}' for event '{reg_event.__name__}'.")
        else:
            self.logger.debug(f"System '{system_func.__name__}' is a vanilla system, not registered for any trigger.")

    async def execute_stage(self, stage: SystemStage, transaction: EcsTransaction) -> None:
        self.logger.info(f"Executing stage: {stage.name} for world: {self.world.name}")
        systems_to_run = self.system_registry.get(stage, [])
        if not systems_to_run:
            self.logger.info(f"No systems registered for stage {stage.name} in world {self.world.name}")
            return

        generators = [
            self._execute_system_func(system_func, transaction, event_object=None, command_object=None)
            for system_func in systems_to_run
        ]
        executor: SystemExecutor = SystemExecutor(generators, ExecutionStrategy.SERIAL)
        async for _ in executor:
            pass

    async def dispatch_event(self, event: BaseEvent, transaction: EcsTransaction) -> None:
        event_type = type(event)
        self.logger.info(f"Dispatching event: {event_type.__name__} for world: {self.world.name}")
        handlers_to_run = self.event_handler_registry.get(event_type, [])
        if not handlers_to_run:
            self.logger.info(
                f"No event handlers registered for event type {event_type.__name__} in world {self.world.name}"
            )
            return

        generators = [
            self._execute_system_func(handler_func, transaction, event_object=event, command_object=None)
            for handler_func in handlers_to_run
        ]
        executor: SystemExecutor = SystemExecutor(generators, ExecutionStrategy.SERIAL)
        async for _ in executor:
            pass

    def dispatch_command(
        self, command: BaseCommand[ResultType], transaction: EcsTransaction
    ) -> SystemExecutor[ResultType]:
        command_type = type(command)
        self.logger.info(f"Dispatching command: {command_type.__name__} for world: {self.world.name}")
        handlers_to_run = self.command_handler_registry.get(command_type, [])

        if not handlers_to_run:
            self.logger.info(
                f"No command handlers registered for command type {command_type.__name__} in world {self.world.name}"
            )
            return SystemExecutor([], command.execution_strategy)

        generators = [
            self._execute_system_func(handler_func, transaction, event_object=None, command_object=command)
            for handler_func in handlers_to_run
        ]
        return SystemExecutor(generators, command.execution_strategy)

    async def run_all_stages(self, transaction: EcsTransaction) -> None:
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

        assert isinstance(metadata, dict)
        params = metadata.get("params")
        if isinstance(params, dict):
            for param_name, value in additional_kwargs.items():
                if param_name in params:
                    kwargs_to_inject[param_name] = value
                else:
                    self.logger.warning(
                        f"Provided kwarg '{param_name}' for system {system_func.__name__} does not match any system parameter. It will be ignored."
                    )

        if params:
            for param_name, param_meta in params.items():
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
    ) -> AsyncGenerator[BaseSystemEvent, None]:
        metadata = self.system_metadata.get(system_func)
        is_async_func = inspect.iscoroutinefunction(system_func)
        if metadata:
            is_async_func = metadata["is_async"]

        kwargs_to_inject: Dict[str, Any] = {}
        try:
            kwargs_to_inject = await self._resolve_dependencies(
                system_func, transaction, event_object, command_object, **additional_kwargs
            )
            self.logger.debug(
                f"Executing system: {system_func.__name__} in world '{self.world.name}' with args: {list(kwargs_to_inject.keys())}"
            )

            result: Any = None
            if is_async_func:
                result = await system_func(**kwargs_to_inject)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: system_func(**kwargs_to_inject))

            if inspect.isasyncgen(result):
                async for item in result:
                    if not isinstance(item, BaseSystemEvent):
                        self.logger.warning(
                            f"System {system_func.__name__} yielded a non-event item of type {type(item)}. "
                            "This may cause issues for consumers expecting BaseSystemEvent subclasses."
                        )
                    yield item
            else:
                yield SystemResultEvent(result)

        finally:
            # This code runs after the generator is exhausted by the consumer.
            if metadata and metadata.get("system_type") == "stage_system":
                if metadata.get("params"):
                    for param_name, param_meta in metadata["params"].items():
                        if param_meta.get("identity") == "MarkedEntityList" and metadata.get(
                            "auto_remove_marker", True
                        ):
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

    def execute_one_time_system(
        self, system_func: Callable[..., Any], transaction: EcsTransaction, **kwargs: Any
    ) -> SystemExecutor[Any]:
        self.logger.info(
            f"Executing one-time system: {system_func.__name__} in world '{self.world.name}' with provided kwargs: {kwargs}"
        )
        generator = self._execute_system_func(
            system_func, transaction, event_object=None, command_object=None, **kwargs
        )
        return SystemExecutor([generator], ExecutionStrategy.SERIAL)
