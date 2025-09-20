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
    cast,
    get_args,
    get_origin,
)

from dam.commands.core import BaseCommand, EventType, ResultType
from dam.core.enums import SystemType
from dam.core.events import BaseEvent
from dam.core.executor import SystemExecutor
from dam.core.markers import CommandMarker, EventMarker, MarkedEntityList, ResourceMarker
from dam.core.resources import ResourceNotFoundError
from dam.core.stages import SystemStage
from dam.core.system_info import SystemMetadata, SystemParameterInfo
from dam.core.transaction import EcsTransaction
from dam.enums import ExecutionStrategy
from dam.models.core.base_component import BaseComponent
from dam.models.core.entity import Entity
from dam.system_events import BaseSystemEvent, SystemResultEvent

if TYPE_CHECKING:
    from dam.core.world import World


SYSTEM_METADATA: Dict[Callable[..., Any], SystemMetadata] = {}
logger = logging.getLogger(__name__)

T = TypeVar("T")


def _parse_system_params(func: Callable[..., Any]) -> Dict[str, SystemParameterInfo]:
    sig = inspect.signature(func)
    param_info: Dict[str, SystemParameterInfo] = {}
    for name, param in sig.parameters.items():
        original_param_type = param.annotation
        identity: Optional[Type[Any]] = None
        actual_type = original_param_type
        marker_component_type: Optional[Type[BaseComponent]] = None
        event_specific_type: Optional[Type[BaseEvent]] = None
        command_specific_type: Optional[Type[BaseCommand[Any, Any]]] = None
        is_annotated = get_origin(original_param_type) is Annotated

        if is_annotated:
            annotated_args = get_args(original_param_type)
            actual_type = annotated_args[0]

            type_based_markers = [arg for arg in annotated_args[1:] if inspect.isclass(arg)]
            identity = next(
                (
                    marker
                    for marker in type_based_markers
                    if marker in [CommandMarker, EventMarker, MarkedEntityList, ResourceMarker]
                ),
                None,
            )

            if not identity:
                string_identities = [arg for arg in annotated_args[1:] if isinstance(arg, str)]
                if string_identities:
                    str_identity = string_identities[0]
                    if str_identity == "Resource":
                        identity = ResourceMarker
                    elif str_identity == "Event":
                        identity = EventMarker
                    elif str_identity == "Command":
                        identity = CommandMarker
                    elif str_identity == "MarkedEntityList":
                        identity = MarkedEntityList

            if identity is MarkedEntityList:
                marker_component_type = next((m for m in type_based_markers if issubclass(m, BaseComponent)), None)
            elif identity is EventMarker:
                if inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
                    event_specific_type = actual_type
            elif identity is CommandMarker:
                if inspect.isclass(actual_type) and issubclass(actual_type, BaseCommand):
                    command_specific_type = actual_type

        if not identity:
            if inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
                identity = EventMarker
                event_specific_type = actual_type
            elif inspect.isclass(actual_type) and issubclass(actual_type, BaseCommand):
                identity = CommandMarker
                command_specific_type = actual_type

        param_info[name] = SystemParameterInfo(
            name=name,
            type_hint=actual_type,
            identity=identity,
            marker_component_type=marker_component_type,
            event_type_hint=event_specific_type,
            command_type_hint=command_specific_type,
            is_annotated=is_annotated,
            original_annotation=original_param_type,
        )
    return param_info


def system(
    func: Optional[Callable[..., Any]] = None,
    *,
    on_stage: Optional[SystemStage] = None,
    on_command: Optional[Type[BaseCommand[Any, Any]]] = None,
    on_event: Optional[Type[BaseEvent]] = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        param_info = _parse_system_params(f)

        system_type = SystemType.VANILLA
        if on_stage:
            system_type = SystemType.STAGE
        elif on_command:
            system_type = SystemType.COMMAND
        elif on_event:
            system_type = SystemType.EVENT

        SYSTEM_METADATA[f] = SystemMetadata(
            func=f,
            params=param_info,
            is_async=inspect.iscoroutinefunction(f),
            system_type=system_type,
            stage=on_stage,
            handles_command_type=on_command,
            listens_for_event_type=on_event,
        )
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
        self.command_handler_registry: Dict[Type[BaseCommand[Any, Any]], List[Callable[..., Any]]] = defaultdict(list)
        self.system_metadata = SYSTEM_METADATA
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def register_system_for_world(
        self,
        system_func: Callable[..., Any],
        stage: Optional[SystemStage] = None,
        event_type: Optional[Type[BaseEvent]] = None,
        command_type: Optional[Type[BaseCommand[Any, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        metadata = self.system_metadata.get(system_func)
        if not metadata:
            return

        if metadata.system_type == SystemType.STAGE and metadata.stage:
            self.system_registry[metadata.stage].append(system_func)
        elif metadata.system_type == SystemType.COMMAND and metadata.handles_command_type:
            reg_command = metadata.handles_command_type
            command_result_type = None
            command_event_type = None
            if reg_command:
                for base in getattr(reg_command, "__orig_bases__", []):
                    if get_origin(base) is BaseCommand:
                        args = get_args(base)
                        if args:
                            command_result_type = args[0]
                        if len(args) > 1:
                            command_event_type = args[1]
                        break

            sig = inspect.signature(system_func)
            handler_return_type = sig.return_annotation

            if handler_return_type is inspect.Signature.empty:
                handler_return_type = Any

            if get_origin(handler_return_type) is AsyncGenerator:
                if command_event_type is not None:
                    handler_event_type = get_args(handler_return_type)[0]
                    if handler_event_type is not Any and handler_event_type != command_event_type:
                        self.logger.warning(
                            f"Potential event type mismatch for command '{reg_command.__name__}'. "
                            f"Handler '{system_func.__name__}' yields '{handler_event_type}' "
                            f"but command expects event type '{command_event_type}'."
                        )
            else:
                if command_result_type is not None and command_result_type is not Any:
                    normalized_command_type = get_origin(command_result_type) or command_result_type
                    normalized_handler_type = get_origin(handler_return_type) or handler_return_type

                    if normalized_handler_type is None:
                        normalized_handler_type = type(None)

                    if normalized_command_type != normalized_handler_type:
                        self.logger.warning(
                            f"Potential return type mismatch for command '{reg_command.__name__}'. "
                            f"Handler '{system_func.__name__}' is annotated to return '{handler_return_type}' "
                            f"but command expects '{command_result_type}'."
                        )
            self.command_handler_registry[reg_command].append(system_func)
        elif metadata.system_type == SystemType.EVENT and metadata.listens_for_event_type:
            self.event_handler_registry[metadata.listens_for_event_type].append(system_func)

    async def execute_stage(self, stage: SystemStage, transaction: EcsTransaction) -> None:
        systems_to_run = self.system_registry.get(stage, [])
        if not systems_to_run:
            return
        generators = [self._execute_system_func(s, transaction) for s in systems_to_run]
        executor = SystemExecutor[Any, BaseSystemEvent](generators, ExecutionStrategy.SERIAL)
        async for _ in executor:
            pass

    async def dispatch_event(self, event: BaseEvent, transaction: EcsTransaction) -> None:
        handlers_to_run = self.event_handler_registry.get(type(event), [])
        if not handlers_to_run:
            return
        generators = [self._execute_system_func(h, transaction, event_object=event) for h in handlers_to_run]
        executor = SystemExecutor[Any, BaseSystemEvent](generators, ExecutionStrategy.SERIAL)
        async for _ in executor:
            pass

    def dispatch_command(
        self, command: BaseCommand[ResultType, EventType], transaction: EcsTransaction
    ) -> SystemExecutor[ResultType, EventType]:
        handlers_to_run = self.command_handler_registry.get(type(command), [])
        if not handlers_to_run:
            return SystemExecutor([], command.execution_strategy)
        generators = [self._execute_system_func(h, transaction, command_object=command) for h in handlers_to_run]
        return SystemExecutor(cast(List[AsyncGenerator[EventType, None]], generators), command.execution_strategy)

    async def _resolve_dependencies(
        self,
        system_func: Callable[..., Any],
        transaction: EcsTransaction,
        event_object: Optional[BaseEvent] = None,
        command_object: Optional[BaseCommand[Any, Any]] = None,
        **additional_kwargs: Any,
    ) -> Dict[str, Any]:
        metadata = self.system_metadata[system_func]
        kwargs_to_inject: Dict[str, Any] = {}

        for param_name, value in additional_kwargs.items():
            if param_name in metadata.params:
                kwargs_to_inject[param_name] = value

        for param_name, param_meta in metadata.params.items():
            if param_name in kwargs_to_inject:
                continue

            identity = param_meta.identity
            param_type_hint = param_meta.type_hint

            if param_type_hint is EcsTransaction:
                kwargs_to_inject[param_name] = transaction
            elif identity is ResourceMarker:
                kwargs_to_inject[param_name] = self.resource_manager.get_resource(param_type_hint)
            elif identity is MarkedEntityList:
                marker_type = param_meta.marker_component_type
                if not marker_type or not issubclass(marker_type, BaseComponent):
                    msg = f"System {system_func.__name__} has MarkedEntityList parameter '{param_name}' with invalid or missing marker component type in world '{self.world.name}'."
                    raise ValueError(msg)
                from sqlalchemy import exists as sql_exists
                from sqlalchemy import select as sql_select

                stmt = sql_select(Entity).where(sql_exists().where(marker_type.entity_id == Entity.id))
                result = await transaction.session.execute(stmt)
                entities_to_process = result.scalars().all()
                kwargs_to_inject[param_name] = entities_to_process
            elif identity is EventMarker:
                if event_object and isinstance(event_object, param_meta.event_type_hint or BaseEvent):
                    kwargs_to_inject[param_name] = event_object
            elif identity is CommandMarker:
                if command_object and isinstance(command_object, param_meta.command_type_hint or BaseCommand):
                    kwargs_to_inject[param_name] = command_object
            else:
                try:
                    kwargs_to_inject[param_name] = self.resource_manager.get_resource(param_type_hint)
                except ResourceNotFoundError:
                    pass
        return kwargs_to_inject

    async def _execute_system_func(
        self,
        system_func: Callable[..., Any],
        transaction: EcsTransaction,
        event_object: Optional[BaseEvent] = None,
        command_object: Optional[BaseCommand[Any, Any]] = None,
        **additional_kwargs: Any,
    ) -> AsyncGenerator[BaseSystemEvent, None]:
        metadata = self.system_metadata.get(system_func)
        if not metadata:
            metadata = SystemMetadata(
                func=system_func,
                params=_parse_system_params(system_func),
                is_async=inspect.iscoroutinefunction(system_func),
                system_type=SystemType.VANILLA,
                stage=None,
                handles_command_type=None,
                listens_for_event_type=None,
            )

        kwargs_to_inject = await self._resolve_dependencies(
            system_func, transaction, event_object, command_object, **additional_kwargs
        )

        result: Any = None
        if metadata.is_async:
            result = await system_func(**kwargs_to_inject)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: system_func(**kwargs_to_inject))

        if inspect.isasyncgen(result):
            async for item in result:
                yield item
        else:
            yield SystemResultEvent(result)

    def execute_one_time_system(
        self, system_func: Callable[..., Any], transaction: EcsTransaction, **kwargs: Any
    ) -> SystemExecutor[Any, BaseSystemEvent]:
        generator = self._execute_system_func(system_func, transaction, **kwargs)
        return SystemExecutor([generator], ExecutionStrategy.SERIAL)
