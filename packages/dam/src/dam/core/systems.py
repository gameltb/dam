"""Core components for the Entity Component System (ECS) framework."""

# pyright: basic
import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Optional,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

from dam.commands.core import BaseCommand
from dam.core.enums import SystemType
from dam.core.events import BaseEvent
from dam.core.executor import SystemExecutor
from dam.core.markers import CommandMarker, EventMarker, MarkedEntityList, ResourceMarker
from dam.core.stages import SystemStage
from dam.core.system_info import SystemMetadata, SystemParameterInfo
from dam.enums import ExecutionStrategy
from dam.models.core.base_component import BaseComponent
from dam.system_events.base import BaseSystemEvent, SystemResultEvent

if TYPE_CHECKING:
    from dam.commands.core import EventType, ResultType
    from dam.core.world import World


SYSTEM_METADATA: dict[Callable[..., Any], SystemMetadata] = {}
logger = logging.getLogger(__name__)

T = TypeVar("T")


def _get_identity_from_str(identity_str: str) -> type[Any] | None:
    """Get marker identity from a string annotation."""
    return {
        "Resource": ResourceMarker,
        "Event": EventMarker,
        "Command": CommandMarker,
        "MarkedEntityList": MarkedEntityList,
    }.get(identity_str)


def _parse_annotated_param(
    annotated_args: tuple[Any, ...],
) -> tuple[type[Any], type[Any] | None, type[BaseComponent] | None]:
    """Parse an Annotated type hint to extract identity and marker component type."""
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
    marker_component_type = None

    if not identity:
        string_identities = [arg for arg in annotated_args[1:] if isinstance(arg, str)]
        if string_identities:
            identity = _get_identity_from_str(string_identities[0])

    if identity is MarkedEntityList:
        marker_component_type = next((m for m in type_based_markers if issubclass(m, BaseComponent)), None)

    return actual_type, identity, marker_component_type


def _parse_system_params(func: Callable[..., Any]) -> dict[str, SystemParameterInfo]:
    """Parse a system function's parameters to extract metadata."""
    sig = inspect.signature(func)
    param_info: dict[str, SystemParameterInfo] = {}
    for name, param in sig.parameters.items():
        original_param_type = param.annotation
        actual_type = original_param_type
        identity: type[Any] | None = None
        marker_component_type: type[BaseComponent] | None = None
        is_annotated = get_origin(original_param_type) is Annotated

        if is_annotated:
            annotated_args = get_args(original_param_type)
            actual_type, identity, marker_component_type = _parse_annotated_param(annotated_args)

        if not identity:
            if inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
                identity = EventMarker
            elif inspect.isclass(actual_type) and issubclass(actual_type, BaseCommand):
                identity = CommandMarker

        event_specific_type: type[BaseEvent] | None = None
        if identity is EventMarker and inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
            event_specific_type = actual_type

        command_specific_type: type[BaseCommand[Any, Any]] | None = None
        if identity is CommandMarker and inspect.isclass(actual_type) and issubclass(actual_type, BaseCommand):
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
    func: Callable[..., Any] | None = None,
    *,
    on_stage: SystemStage | None = None,
    on_command: type["BaseCommand[Any, Any]"] | None = None,
    on_event: type[BaseEvent] | None = None,
    **_kwargs: Any,
) -> Callable[..., Any]:
    """Register a function as a system."""

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
    """Schedules and executes systems within a world."""

    def __init__(self, world: "World") -> None:
        """Initialize the scheduler."""
        self.world = world
        self.resource_manager = world.resource_manager
        self.system_registry: dict[SystemStage, list[Callable[..., Any]]] = defaultdict(list)
        self.event_handler_registry: dict[type[BaseEvent], list[Callable[..., Any]]] = defaultdict(list)
        self.command_handler_registry: dict[type[BaseCommand[Any, Any]], list[Callable[..., Any]]] = defaultdict(list)
        self.system_metadata = SYSTEM_METADATA
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _validate_command_handler_signature(
        self, system_func: Callable[..., Any], reg_command: type["BaseCommand[Any, Any]"]
    ) -> None:
        """Validate the signature of a command handler against the command definition."""
        command_result_type, command_event_type = None, None
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
                        "Potential event type mismatch for command '%s'. Handler '%s' yields '%s' but command expects '%s'.",
                        reg_command.__name__,
                        system_func.__name__,
                        handler_event_type,
                        command_event_type,
                    )
        elif command_result_type is not None and command_result_type is not Any:
            normalized_command_type = get_origin(command_result_type) or command_result_type
            normalized_handler_type = get_origin(handler_return_type) or handler_return_type
            if normalized_handler_type is None:
                normalized_handler_type = type(None)
            if normalized_command_type != normalized_handler_type:
                self.logger.warning(
                    "Potential return type mismatch for command '%s'. Handler '%s' is annotated to return '%s' but command expects '%s'.",
                    reg_command.__name__,
                    system_func.__name__,
                    handler_return_type,
                    command_result_type,
                )

    def _register_command_handler(
        self, system_func: Callable[..., Any], reg_command: type["BaseCommand[Any, Any]"]
    ) -> None:
        """Register a command handler and validate its signature."""
        self._validate_command_handler_signature(system_func, reg_command)
        self.command_handler_registry[reg_command].append(system_func)

    def register_system_for_world(
        self,
        system_func: Callable[..., Any],
        _stage: SystemStage | None = None,
        _event_type: type[BaseEvent] | None = None,
        _command_type: type["BaseCommand[Any, Any]"] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Register a system with the appropriate registry based on its metadata."""
        metadata = self.system_metadata.get(system_func)
        if not metadata:
            return

        if metadata.system_type == SystemType.STAGE and metadata.stage:
            self.system_registry[metadata.stage].append(system_func)
        elif metadata.system_type == SystemType.COMMAND and metadata.handles_command_type:
            self._register_command_handler(system_func, metadata.handles_command_type)
        elif metadata.system_type == SystemType.EVENT and metadata.listens_for_event_type:
            self.event_handler_registry[metadata.listens_for_event_type].append(system_func)

    async def execute_stage(self, stage: SystemStage) -> None:
        """Execute all systems registered for a given stage."""
        systems_to_run = self.system_registry.get(stage, [])
        if not systems_to_run:
            return
        generators = [self._execute_system_func(s) for s in systems_to_run]
        executor = SystemExecutor[Any, BaseSystemEvent](generators, ExecutionStrategy.SERIAL)
        async for _ in executor:
            pass

    async def dispatch_event(self, event: BaseEvent) -> None:
        """Dispatch an event to all registered handlers."""
        handlers_to_run = self.event_handler_registry.get(type(event), [])
        if not handlers_to_run:
            return
        generators = [self._execute_system_func(h, event_object=event) for h in handlers_to_run]
        executor = SystemExecutor[Any, BaseSystemEvent](generators, ExecutionStrategy.SERIAL)
        async for _ in executor:
            pass

    def dispatch_command(
        self, command: "BaseCommand[ResultType, EventType]", **kwargs: Any
    ) -> "SystemExecutor[ResultType, EventType]":
        """Dispatch a command to its registered handlers and return an executor."""
        handlers_to_run = self.command_handler_registry.get(type(command), [])
        if not handlers_to_run:
            self.logger.warning(
                "No system registered for command %s in world '%s'.", type(command).__name__, self.world.name
            )
            return SystemExecutor([], command.execution_strategy)
        generators = [self._execute_system_func(h, command_object=command, **kwargs) for h in handlers_to_run]
        return SystemExecutor(cast(list[AsyncGenerator["EventType", None]], generators), command.execution_strategy)

    def _get_initial_dependencies(
        self,
        metadata: SystemMetadata,
        event_object: BaseEvent | None,
        command_object: Optional["BaseCommand[Any, Any]"],
        additional_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[type, Any]]:
        """Populate initial dependencies from event, command, and kwargs."""
        resolved_params_by_name: dict[str, Any] = {}
        resolved_deps_by_type: dict[type, Any] = {}

        if event_object:
            for name, param in metadata.params.items():
                if param.identity is EventMarker and isinstance(event_object, param.event_type_hint or BaseEvent):
                    resolved_params_by_name[name] = event_object
                    resolved_deps_by_type[param.type_hint] = event_object
                    break
        if command_object:
            for name, param in metadata.params.items():
                if param.identity is CommandMarker and isinstance(command_object, param.command_type_hint or object):
                    resolved_params_by_name[name] = command_object
                    resolved_deps_by_type[param.type_hint] = command_object
                    break
        resolved_params_by_name.update(additional_kwargs)
        for _key, value in additional_kwargs.items():
            resolved_deps_by_type[type(value)] = value

        return resolved_params_by_name, resolved_deps_by_type

    async def _try_resolve_dependency(
        self, stack: AsyncExitStack, param: SystemParameterInfo, resolved_deps_by_type: dict[type, Any]
    ) -> tuple[Any | None, bool]:
        """Attempt to resolve a single dependency using a context provider or resource."""
        provider_key = MarkedEntityList if param.identity is MarkedEntityList else param.type_hint
        provider = self.world.context_providers.get(provider_key)

        if provider:
            provider_sig = inspect.signature(provider.__call__)
            provider_kwargs = {}
            can_resolve = True
            for p_name, p_param in provider_sig.parameters.items():
                if p_name == "self" or p_param.kind in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    continue
                if param.identity is MarkedEntityList and p_name == "marker_component_type":
                    provider_kwargs[p_name] = param.marker_component_type
                    continue
                p_dep_type = p_param.annotation
                if p_dep_type in resolved_deps_by_type:
                    provider_kwargs[p_name] = resolved_deps_by_type[p_dep_type]
                else:
                    can_resolve = False
                    break
            if can_resolve:
                context = cast(AbstractAsyncContextManager, provider(**provider_kwargs))
                resolved_value = await stack.enter_async_context(context)
                return resolved_value, True
            return None, False

        if self.resource_manager.has_resource(param.type_hint):
            return self.resource_manager.get_resource(param.type_hint), True

        return None, False

    async def _resolve_system_dependencies(
        self,
        stack: AsyncExitStack,
        system_func: Callable[..., Any],
        metadata: SystemMetadata,
        resolved_params_by_name: dict[str, Any],
        resolved_deps_by_type: dict[type, Any],
    ) -> None:
        """Resolve all dependencies for a system iteratively."""
        unresolved_params = {
            name: meta for name, meta in metadata.params.items() if name not in resolved_params_by_name
        }
        for _ in range(len(unresolved_params) + 1):
            if not unresolved_params:
                break
            newly_resolved: dict[str, Any] = {}
            still_unresolved: dict[str, SystemParameterInfo] = {}
            for name, param in unresolved_params.items():
                resolved_value, is_resolved = await self._try_resolve_dependency(stack, param, resolved_deps_by_type)
                if is_resolved:
                    newly_resolved[name] = resolved_value
                    resolved_deps_by_type[param.type_hint] = resolved_value
                else:
                    still_unresolved[name] = param
            if not newly_resolved and still_unresolved:
                unresolved_names = ", ".join(still_unresolved.keys())
                raise RuntimeError(
                    f"Could not resolve dependencies for system '{system_func.__name__}': {unresolved_names}"
                )
            resolved_params_by_name.update(newly_resolved)
            unresolved_params = still_unresolved

    async def _execute_system_func(
        self,
        system_func: Callable[..., Any],
        event_object: BaseEvent | None = None,
        command_object: Optional["BaseCommand[Any, Any]"] = None,
        **additional_kwargs: Any,
    ) -> AsyncGenerator[BaseSystemEvent, None]:
        """Prepare and execute a system function, resolving all its dependencies."""
        metadata = self.system_metadata.get(system_func) or SystemMetadata(
            func=system_func,
            params=_parse_system_params(system_func),
            is_async=inspect.iscoroutinefunction(system_func),
            system_type=SystemType.VANILLA,
            stage=None,
            handles_command_type=None,
            listens_for_event_type=None,
        )

        gexit_raised = False
        try:
            async with AsyncExitStack() as stack:
                resolved_params, resolved_types = self._get_initial_dependencies(
                    metadata, event_object, command_object, additional_kwargs
                )
                await self._resolve_system_dependencies(stack, system_func, metadata, resolved_params, resolved_types)

                result: Any
                if metadata.is_async:
                    result = await system_func(**resolved_params)
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, lambda: system_func(**resolved_params))

                try:
                    if inspect.isasyncgen(result):
                        async for item in result:
                            yield item
                    else:
                        yield SystemResultEvent(result=result)
                except GeneratorExit:
                    gexit_raised = True
        finally:
            if gexit_raised:
                raise GeneratorExit()

    def execute_one_time_system(
        self, system_func: Callable[..., Any], **kwargs: Any
    ) -> SystemExecutor[Any, BaseSystemEvent]:
        """Execute a single system function immediately, outside the normal flow."""
        generator = self._execute_system_func(system_func, **kwargs)
        return SystemExecutor([generator], ExecutionStrategy.SERIAL)
