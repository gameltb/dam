import asyncio
import inspect
import logging
from collections import defaultdict
from typing import Annotated, Any, Callable, Dict, List, Optional, Type, get_args, get_origin

from dam.core.events import BaseEvent
from dam.core.exceptions import EventHandlingError, StageExecutionError
from dam.core.resources import ResourceManager, ResourceNotFoundError
from dam.core.stages import SystemStage
from dam.core.system_params import WorldContext
from dam.models import BaseComponent, Entity

SYSTEM_METADATA: Dict[Callable[..., Any], Dict[str, Any]] = {}
logger = logging.getLogger(__name__)


def _parse_system_params(func: Callable[..., Any]) -> Dict[str, Any]:
    sig = inspect.signature(func)
    param_info = {}
    for name, param in sig.parameters.items():
        original_param_type = param.annotation
        identity: Optional[str] = None
        actual_type = original_param_type
        marker_component_type: Optional[Type[BaseComponent]] = None
        event_specific_type: Optional[Type[BaseEvent]] = None

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
                        f"Parameter '{name}' in system '{func.__name__}' is 'MarkedEntityList' but "
                        f"is missing a BaseComponent subclass in Annotated args."
                    )
            elif identity == "Event":
                if inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
                    event_specific_type = actual_type
                else:
                    logger.warning(
                        f"Parameter '{name}' in system '{func.__name__}' is 'Event' but its type "
                        f"'{actual_type}' is not a BaseEvent subclass."
                    )

        if not identity:
            if actual_type is WorldContext:
                identity = "WorldContext"
            elif inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
                identity = "Event"
                event_specific_type = actual_type

        if identity == "Event" and not event_specific_type:
            if inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
                event_specific_type = actual_type
            else:
                logger.warning(
                    f"Parameter '{name}' in system '{func.__name__}' resolved to 'Event' identity, "
                    f"but its type '{actual_type}' is not a BaseEvent subclass."
                )

        param_info[name] = {
            "name": name,
            "type_hint": actual_type,
            "identity": identity,
            "marker_component_type": marker_component_type,
            "event_type_hint": event_specific_type,
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
                    f"System {func.__name__} registered for event {event_type.__name__} but does not "
                    f"seem to have a parameter matching this event type."
                )
        return func

    return decorator


class WorldScheduler:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.system_registry: Dict[SystemStage, List[Callable[..., Any]]] = defaultdict(list)
        self.event_handler_registry: Dict[Type[BaseEvent], List[Callable[..., Any]]] = defaultdict(list)
        self.system_metadata = SYSTEM_METADATA
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def register_system_for_world(
        self,
        system_func: Callable[..., Any],
        stage: Optional[SystemStage] = None,
        event_type: Optional[Type[BaseEvent]] = None,
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
            self.logger.info(f"System {system_func.__name__} registered for stage {stage.name} in this scheduler.")
        elif event_type:
            self.event_handler_registry[event_type].append(system_func)
            self.logger.info(
                f"System {system_func.__name__} registered for event {event_type.__name__} in this scheduler."
            )
        else:
            self.logger.error(f"System {system_func.__name__} must be registered with either a stage or an event_type.")

    async def _resolve_and_execute_system(
        self, system_func: Callable[..., Any], world_context: WorldContext, event_object: Optional[BaseEvent] = None
    ):
        metadata = self.system_metadata.get(system_func)
        if not metadata:
            self.logger.warning(
                f"No metadata found for system {system_func.__name__} in world '{world_context.world_name}'. Skipping."
            )
            return False

        kwargs_to_inject = {}
        try:
            for param_name, param_meta in metadata["params"].items():
                identity = param_meta["identity"]
                param_type_hint = param_meta["type_hint"]

                if identity == "WorldSession":
                    kwargs_to_inject[param_name] = world_context.session
                elif identity == "WorldName":
                    kwargs_to_inject[param_name] = world_context.world_name
                elif identity == "CurrentWorldConfig":
                    kwargs_to_inject[param_name] = world_context.world_config
                elif identity == "WorldContext":
                    kwargs_to_inject[param_name] = world_context
                elif identity == "Resource":
                    try:
                        kwargs_to_inject[param_name] = self.resource_manager.get_resource(param_type_hint)
                    except ResourceNotFoundError as e:
                        self.logger.error(
                            f"System {system_func.__name__} in world '{world_context.world_name}' requires resource "
                            f"{param_type_hint.__name__} which was not found: {e}",
                            exc_info=True,
                        )
                        raise
                elif identity == "MarkedEntityList":
                    marker_type = param_meta["marker_component_type"]
                    if not marker_type or not issubclass(marker_type, BaseComponent):
                        msg = (
                            f"System {system_func.__name__} has MarkedEntityList parameter '{param_name}' "
                            f"with invalid or missing marker component type in world '{world_context.world_name}'."
                        )
                        self.logger.error(msg)
                        raise ValueError(msg)

                    from sqlalchemy import exists as sql_exists
                    from sqlalchemy import select as sql_select

                    stmt = sql_select(Entity).where(sql_exists().where(marker_type.entity_id == Entity.id))
                    entities_to_process = world_context.session.execute(stmt).scalars().all()
                    kwargs_to_inject[param_name] = entities_to_process
                    self.logger.debug(
                        f"System {system_func.__name__} in world '{world_context.world_name}' gets {len(entities_to_process)} entities for marker {marker_type.__name__}"
                    )
                elif identity == "Event":
                    expected_event_type = param_meta["event_type_hint"]
                    if event_object and isinstance(event_object, expected_event_type):
                        kwargs_to_inject[param_name] = event_object
                    elif event_object:
                        self.logger.warning(
                            f"System {system_func.__name__} in world '{world_context.world_name}' expected event type {expected_event_type} "
                            f"but received {type(event_object)}. Skipping injection for this param."
                        )
                    elif expected_event_type is not None:
                        msg = (
                            f"System {system_func.__name__} parameter '{param_name}' in world '{world_context.world_name}' "
                            f"expects an event of type {expected_event_type.__name__} but none was provided."
                        )
                        self.logger.error(msg)
                        raise ValueError(msg)
                else:  # No specific identity, try direct resource injection
                    # Check if it's the event_object itself, which should only be injected if identity is "Event"
                    is_event_param_for_current_event = False
                    if event_object and param_meta.get("event_type_hint"):  # Check if param expects an event
                        if isinstance(
                            event_object, param_meta["event_type_hint"]
                        ):  # Check if current event matches param's expected event type
                            # Check if this param_name is the one designated for this event type via @listens_for
                            # This is a bit heuristic: find the param that's typed as the event this handler is for.
                            event_param_candidate_name = None
                            listens_for_type = metadata.get("listens_for_event_type")  # Get from system metadata
                            if listens_for_type:
                                for pn, pi in metadata["params"].items():
                                    if pi.get("type_hint") == listens_for_type and pi.get("identity") == "Event":
                                        event_param_candidate_name = pn
                                        break
                            if param_name == event_param_candidate_name:
                                is_event_param_for_current_event = True

                    if not is_event_param_for_current_event:
                        # Avoid basic types and NoneType
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
                            self.logger.debug(
                                f"No specific identity for param '{param_name}'. Attempting direct resource injection for type: {param_type_hint}"
                            )
                            try:
                                resource_instance = self.resource_manager.get_resource(param_type_hint)
                                kwargs_to_inject[param_name] = resource_instance
                                self.logger.debug(
                                    f"Successfully injected resource for param: {param_name} by direct type: {param_type_hint}"
                                )
                            except ResourceNotFoundError:
                                self.logger.warning(
                                    f"Resource not found for param: {param_name} by direct type: {param_type_hint}. This may lead to a TypeError if the parameter is required."
                                )
                            except Exception as e_direct_inject:
                                self.logger.error(
                                    f"Error during direct resource injection attempt for param '{param_name}' (type {param_type_hint}): {e_direct_inject}",
                                    exc_info=True,
                                )
        except Exception as e:
            self.logger.error(
                f"Error preparing dependencies for system {system_func.__name__} in world '{world_context.world_name}': {e}",
                exc_info=True,
            )
            raise

        self.logger.debug(
            f"Executing system: {system_func.__name__} in world '{world_context.world_name}' with args: {list(kwargs_to_inject.keys())}"
        )

        if metadata["is_async"]:
            await system_func(**kwargs_to_inject)
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: system_func(**kwargs_to_inject))

        if metadata.get("system_type") == "stage_system":
            for param_name, param_meta in metadata["params"].items():
                if param_meta.get("identity") == "MarkedEntityList" and metadata.get("auto_remove_marker", True):
                    marker_type_to_remove = param_meta["marker_component_type"]
                    entities_processed = kwargs_to_inject.get(param_name, [])
                    if entities_processed and marker_type_to_remove:
                        entity_ids_processed = [entity.id for entity in entities_processed]
                        if entity_ids_processed:
                            from sqlalchemy import delete as sql_delete

                            self.logger.debug(
                                f"Scheduler for world '{world_context.world_name}': Bulk removing {marker_type_to_remove.__name__} from "
                                f"{len(entity_ids_processed)} entities after system {system_func.__name__}."
                            )
                            stmt = sql_delete(marker_type_to_remove).where(
                                marker_type_to_remove.entity_id.in_(entity_ids_processed)
                            )
                            world_context.session.execute(stmt)
                            world_context.session.flush()
        return True

    async def execute_stage(self, stage: SystemStage, world_context: WorldContext):
        self.logger.info(f"Executing stage: {stage.name} for world: {world_context.world_name}")
        systems_to_run = self.system_registry.get(stage, [])
        if not systems_to_run:
            self.logger.info(f"No systems registered for stage {stage.name} in world {world_context.world_name}")
            return

        active_system_func_name = "None"
        try:
            for system_func in systems_to_run:
                active_system_func_name = system_func.__name__
                await self._resolve_and_execute_system(system_func, world_context)
            try:
                world_context.session.commit()
                self.logger.info(f"Committed session for stage {stage.name} in world {world_context.world_name}")
            except Exception as commit_exc:
                self.logger.error(
                    f"Error committing session for stage {stage.name} in world {world_context.world_name}: {commit_exc}. Rolling back.",
                    exc_info=True,
                )
                world_context.session.rollback()
                raise StageExecutionError(
                    message=f"Failed to commit stage {stage.name} in world {world_context.world_name}.",
                    stage_name=stage.name,
                    original_exception=commit_exc,
                ) from commit_exc
        except Exception as system_exc:
            self.logger.error(
                f"System '{active_system_func_name}' failed in stage '{stage.name}' for world '{world_context.world_name}'. Rolling back. Error: {system_exc}",
                exc_info=True,
            )
            world_context.session.rollback()
            raise StageExecutionError(
                message=f"System '{active_system_func_name}' failed during stage '{stage.name}' execution in world '{world_context.world_name}'.",
                stage_name=stage.name,
                system_name=active_system_func_name,
                original_exception=system_exc,
            ) from system_exc

    async def dispatch_event(self, event: BaseEvent, world_context: WorldContext):
        event_type = type(event)
        self.logger.info(f"Dispatching event: {event_type.__name__} for world: {world_context.world_name}")
        handlers_to_run = self.event_handler_registry.get(event_type, [])
        if not handlers_to_run:
            self.logger.info(
                f"No event handlers registered for event type {event_type.__name__} in world {world_context.world_name}"
            )
            return

        active_handler_func_name = "None"
        try:
            for handler_func in handlers_to_run:
                active_handler_func_name = handler_func.__name__
                await self._resolve_and_execute_system(handler_func, world_context, event_object=event)
            try:
                world_context.session.commit()
                self.logger.info(
                    f"Committed session after handling event {event_type.__name__} in world {world_context.world_name}"
                )
            except Exception as commit_exc:
                self.logger.error(
                    f"Error committing session after event {event_type.__name__} in world {world_context.world_name}: {commit_exc}. Rolling back.",
                    exc_info=True,
                )
                world_context.session.rollback()
                raise EventHandlingError(
                    message=f"Failed to commit after handling event {event_type.__name__} in world {world_context.world_name}.",
                    event_type=event_type.__name__,
                    original_exception=commit_exc,
                ) from commit_exc
        except Exception as handler_exc:
            self.logger.error(
                f"Handler '{active_handler_func_name}' failed for event '{event_type.__name__}' in world '{world_context.world_name}'. Rolling back. Error: {handler_exc}",
                exc_info=True,
            )
            world_context.session.rollback()
            raise EventHandlingError(
                message=f"Handler '{active_handler_func_name}' failed for event '{event_type.__name__}' in world '{world_context.world_name}'.",
                event_type=event_type.__name__,
                handler_name=active_handler_func_name,
                original_exception=handler_exc,
            ) from handler_exc

    async def run_all_stages(self, initial_world_context: WorldContext):
        self.logger.info(f"Attempting to run all stages for world: {initial_world_context.world_name}")
        ordered_stages = sorted(list(SystemStage), key=lambda s: s.value if isinstance(s.value, int) else str(s.value))
        for stage in ordered_stages:
            self.logger.info(
                f"Running stage {stage.name} as part of run_all_stages for world {initial_world_context.world_name}."
            )
            await self.execute_stage(stage, initial_world_context)
        self.logger.info(f"Finished running all stages for world: {initial_world_context.world_name}")

    async def _resolve_dependencies(
        self,
        system_func: Callable[..., Any],
        world_context: WorldContext,
        event_object: Optional[BaseEvent] = None,
        **additional_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Resolves dependencies for a given system function, incorporating additional kwargs.
        """
        metadata = self.system_metadata.get(system_func)
        if not metadata:
            # Try to parse dynamically if not found (e.g., for one-time systems not using decorators)
            self.logger.warning(
                f"No pre-registered metadata for system {system_func.__name__} in world '{world_context.world_name}'. "
                "Attempting dynamic parameter parsing."
            )
            params_info = _parse_system_params(system_func)
            metadata = {
                "params": params_info,
                "is_async": inspect.iscoroutinefunction(system_func),
                # Add other necessary default metadata if needed, or handle their absence
            }
            # Optionally, store this dynamically generated metadata if it's useful for repeated calls,
            # though for one-time systems, this might not be necessary.
            # self.system_metadata[system_func] = metadata # Be cautious with modifying shared state here

        kwargs_to_inject: Dict[str, Any] = {}

        # First, fill with additional_kwargs provided by the caller
        # This allows overriding or providing specific values for parameters.
        for param_name, value in additional_kwargs.items():
            if param_name in metadata["params"]:
                kwargs_to_inject[param_name] = value
            else:
                self.logger.warning(
                    f"Provided kwarg '{param_name}' for system {system_func.__name__} does not match any system parameter. It will be ignored."
                )

        # Then, resolve remaining dependencies
        for param_name, param_meta in metadata["params"].items():
            if param_name in kwargs_to_inject:  # Already provided by additional_kwargs
                continue

            identity = param_meta["identity"]
            param_type_hint = param_meta["type_hint"]

            if identity == "WorldSession":
                kwargs_to_inject[param_name] = world_context.session
            elif identity == "WorldName":
                kwargs_to_inject[param_name] = world_context.world_name
            elif identity == "CurrentWorldConfig":
                kwargs_to_inject[param_name] = world_context.world_config
            elif identity == "WorldContext":
                kwargs_to_inject[param_name] = world_context
            elif identity == "Resource":
                kwargs_to_inject[param_name] = self.resource_manager.get_resource(param_type_hint)
            elif identity == "MarkedEntityList":
                marker_type = param_meta["marker_component_type"]
                if not marker_type or not issubclass(marker_type, BaseComponent):
                    msg = (
                        f"System {system_func.__name__} has MarkedEntityList parameter '{param_name}' "
                        f"with invalid or missing marker component type in world '{world_context.world_name}'."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                from sqlalchemy import exists as sql_exists
                from sqlalchemy import select as sql_select

                stmt = sql_select(Entity).where(sql_exists().where(marker_type.entity_id == Entity.id))
                entities_to_process = world_context.session.execute(stmt).scalars().all()
                kwargs_to_inject[param_name] = entities_to_process
            elif identity == "Event":
                expected_event_type = param_meta["event_type_hint"]
                if event_object and isinstance(event_object, expected_event_type):
                    kwargs_to_inject[param_name] = event_object
                elif expected_event_type is not None and not event_object:  # Event expected but none given
                    msg = (
                        f"System {system_func.__name__} parameter '{param_name}' in world '{world_context.world_name}' "
                        f"expects an event of type {expected_event_type.__name__} but none was provided for injection."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
            else:  # No specific identity, try direct resource injection if not a basic type
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
        world_context: WorldContext,
        event_object: Optional[BaseEvent] = None,
        **additional_kwargs: Any,
    ):
        """
        Internal helper to execute a system function after resolving its dependencies.
        Incorporates additional_kwargs for flexible execution (e.g., for one-time systems).
        """
        metadata = self.system_metadata.get(system_func)
        # If metadata is not found, _resolve_dependencies will attempt dynamic parsing.
        # We still need to know if it's async.
        is_async_func = inspect.iscoroutinefunction(system_func)
        if metadata:  # if pre-registered, use its async flag
            is_async_func = metadata["is_async"]

        try:
            kwargs_to_inject = await self._resolve_dependencies(
                system_func, world_context, event_object, **additional_kwargs
            )
        except Exception as e:
            self.logger.error(
                f"Error resolving dependencies for system {system_func.__name__} in world '{world_context.world_name}': {e}",
                exc_info=True,
            )
            raise  # Re-raise to be caught by caller (execute_stage, dispatch_event, or execute_one_time_system)

        self.logger.debug(
            f"Executing system: {system_func.__name__} in world '{world_context.world_name}' with args: {list(kwargs_to_inject.keys())}"
        )

        if is_async_func:
            await system_func(**kwargs_to_inject)
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: system_func(**kwargs_to_inject))

        # Auto-removal of marker components is specific to registered stage systems with metadata.
        # One-time systems or event handlers might not use this pattern or expect this behavior by default.
        if metadata and metadata.get("system_type") == "stage_system":
            for param_name, param_meta in metadata["params"].items():
                if param_meta.get("identity") == "MarkedEntityList" and metadata.get("auto_remove_marker", True):
                    marker_type_to_remove = param_meta["marker_component_type"]
                    # Use the injected list from kwargs_to_inject, not a fresh query
                    entities_processed = kwargs_to_inject.get(param_name, [])
                    if entities_processed and marker_type_to_remove:
                        entity_ids_processed = [entity.id for entity in entities_processed]
                        if entity_ids_processed:
                            from sqlalchemy import delete as sql_delete

                            self.logger.debug(
                                f"Scheduler for world '{world_context.world_name}': Bulk removing {marker_type_to_remove.__name__} from "
                                f"{len(entity_ids_processed)} entities after system {system_func.__name__}."
                            )
                            stmt = sql_delete(marker_type_to_remove).where(
                                marker_type_to_remove.entity_id.in_(entity_ids_processed)
                            )
                            world_context.session.execute(stmt)
                            # Flush is important here if subsequent systems in the same stage need to see this change.
                            # For one-time systems, commit/flush is handled by the caller.
                            if metadata.get("system_type") == "stage_system":
                                world_context.session.flush()
        return True  # Indicates successful execution of the function itself

    async def execute_one_time_system(
        self, system_func: Callable[..., Any], world_context: WorldContext, **kwargs: Any
    ):
        """
        Executes a single, dynamically provided system function immediately.
        Dependencies are resolved, and the system is run.
        The caller (World.execute_one_time_system) is responsible for session management (commit/rollback).
        """
        self.logger.info(
            f"Executing one-time system: {system_func.__name__} in world '{world_context.world_name}' with provided kwargs: {kwargs}"
        )
        try:
            await self._execute_system_func(system_func, world_context, event_object=None, **kwargs)
            # For one-time systems, commit is typically handled by the calling context (e.g., World method)
            # If immediate commit is desired here, it would be:
            # world_context.session.commit()
            # self.logger.info(f"Committed session after one-time system {system_func.__name__}")
        except Exception as e:
            self.logger.error(
                f"Error during execution of one-time system {system_func.__name__} in world '{world_context.world_name}': {e}. "
                "Session rollback should be handled by the caller.",
                exc_info=True,
            )
            # world_context.session.rollback() # Caller handles rollback
            raise  # Re-raise the exception to be handled by the World method

    # Overwrite _resolve_and_execute_system to use the new _execute_system_func structure
    async def _resolve_and_execute_system(
        self, system_func: Callable[..., Any], world_context: WorldContext, event_object: Optional[BaseEvent] = None
    ):
        # This method is primarily used by execute_stage and dispatch_event for registered systems.
        # It will call _execute_system_func without additional_kwargs, relying on standard DI.
        return await self._execute_system_func(system_func, world_context, event_object)
