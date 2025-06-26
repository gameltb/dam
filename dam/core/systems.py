import asyncio
import inspect
import logging  # Added logging
from collections import defaultdict
from typing import Annotated, Any, Callable, Dict, List, Optional, Type, get_args, get_origin

from dam.core.events import BaseEvent
from dam.core.exceptions import EventHandlingError, StageExecutionError  # Task 5.1: Import new exceptions
from dam.core.resources import ResourceManager, ResourceNotFoundError  # Assuming this is the one to keep
from dam.core.stages import SystemStage
from dam.core.system_params import (
    WorldContext,
)
from dam.models import BaseComponent, Entity

# --- System Registration and Metadata ---

# Global SYSTEM_METADATA remains, as it stores parameter info by function, which is universal.
# SYSTEM_REGISTRY and EVENT_HANDLER_REGISTRY are now instance variables on WorldScheduler.

SYSTEM_METADATA: Dict[Callable[..., Any], Dict[str, Any]] = {}
"""
Global dictionary storing metadata for each registered system function (stage-based or event-based).
Keys are the system functions themselves.
Values are dictionaries containing parsed parameter information (name, type, identity),
async status, and any other kwargs passed to the decorator.
Example:
{
    my_system_func: {
        "params": {
            "param1_name": {"name": "param1", "type_hint": Session, "identity": "WorldSession", ...},
            "param2_name": {"name": "param2", "type_hint": MyResource, "identity": "Resource", ...},
            "event_param_name": {"name": "event", "type_hint": MyCustomEvent, "identity": "Event", ...}
        },
        "is_async": True,
        "custom_decorator_arg": "value",
        "listens_for_event_type": MyCustomEvent # For event handlers
    }
}
"""


# Need to ensure these are imported for type checking in _parse_system_params
# WorldContext is already imported

# ... (other imports)


def _parse_system_params(func: Callable[..., Any]) -> Dict[str, Any]:
    """Helper function to parse parameters of a system function."""
    sig = inspect.signature(func)
    param_info = {}
    for name, param in sig.parameters.items():
        original_param_type = param.annotation

        identity: Optional[str] = None
        actual_type = original_param_type  # Default actual_type to the original annotation
        marker_component_type: Optional[Type[BaseComponent]] = None
        event_specific_type: Optional[Type[BaseEvent]] = None  # Stores the specific MyEvent for an "Event" identity

        if get_origin(original_param_type) is Annotated:
            annotated_args = get_args(original_param_type)
            actual_type = annotated_args[0]  # The core type, e.g., Session, List[Entity], MyEvent, MyResource

            string_identities_found = []
            type_based_markers_found = []

            for ann_arg in annotated_args[1:]:
                if isinstance(ann_arg, str):
                    string_identities_found.append(ann_arg)
                elif inspect.isclass(ann_arg):
                    type_based_markers_found.append(ann_arg)

            # Prefer string identity if multiple are provided, but log warning.
            if len(string_identities_found) > 1:
                logger.warning(
                    f"Parameter '{name}' in system '{func.__name__}' has multiple string annotations: "
                    f"{string_identities_found}. Using the first one: '{string_identities_found[0]}'."
                )
            if string_identities_found:
                identity = string_identities_found[0]

            if identity == "MarkedEntityList":
                found_marker = False
                for marker in type_based_markers_found:
                    if issubclass(marker, BaseComponent):
                        marker_component_type = marker
                        found_marker = True
                        break
                if not found_marker:
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
            # Other identities like "WorldSession", "CurrentWorldConfig", "Resource", "WorldName"
            # are set directly from string.

        # Handle non-Annotated cases or cases where Annotated didn't set a specific identity
        if not identity:  # No string identity from Annotated, or not Annotated at all
            if actual_type is WorldContext:  # Direct type hint for WorldContext
                identity = "WorldContext"
            elif inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):  # Direct type hint for an Event
                identity = "Event"
                event_specific_type = actual_type
            # For Session, WorldConfig, and generic Resources, explicit Annotated[..., "IdentityString"]
            # is now preferred. This avoids ambiguity if a system uses other types of Session objects or
            # config objects. Example: A system might take Annotated[Session, "WorldSession"] and also
            # another_session: SomeOtherSessionType.

        # Final consistency check for event_specific_type if identity is "Event"
        if identity == "Event" and not event_specific_type:
            if inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
                event_specific_type = actual_type
            else:  # Should have been caught if actual_type was not BaseEvent and identity came from string.
                # This path is more for if identity was set some other way or for future implicit rules.
                logger.warning(
                    f"Parameter '{name}' in system '{func.__name__}' resolved to 'Event' identity, "
                    f"but its type '{actual_type}' is not a BaseEvent subclass."
                )

        param_info[name] = {
            "name": name,
            "type_hint": actual_type,  # The core Python type (e.g., Session, MyResource, List[Entity], MyEvent)
            "identity": identity,  # The string tag (e.g., "WorldSession", "Resource", "Event")
            "marker_component_type": marker_component_type,  # Specific type for MarkedEntityList (e.g., NeedsProcessingComponent)
            "event_type_hint": event_specific_type,  # Specific event class for "Event" (e.g., MyCustomEvent)
            "is_annotated": get_origin(original_param_type) is Annotated,
            "original_annotation": original_param_type,
        }
    return param_info


def system(stage: SystemStage, **kwargs):
    """
    Decorator to register a function as a system to be run in a specific `SystemStage`.

    The decorator introspects the decorated function's parameters, expecting type hints
    (often `typing.Annotated`) to declare dependencies like `WorldSession`,
    `WorldConfig`, specific `Resource` types, or `MarkedEntityList[SomeMarkerComponent]`.
    This information is stored in `SYSTEM_METADATA`.

    Args:
        stage: The `SystemStage` during which this system should run.
        **kwargs: Additional arbitrary keyword arguments that can be stored as metadata
                  for the system (e.g., `auto_remove_marker=False`).
    """

    def decorator(func: Callable[..., Any]):
        # Decorator no longer adds to global SYSTEM_REGISTRY.
        # It primarily populates SYSTEM_METADATA.
        # Registration to a specific world's scheduler happens via world.register_system().
        param_info = _parse_system_params(func)
        SYSTEM_METADATA[func] = {
            "params": param_info,
            "is_async": inspect.iscoroutinefunction(func),
            "system_type": "stage_system",  # Mark as stage-based
            "stage": stage,  # Task 2.1: Store stage in metadata
            **kwargs,
        }
        return func

    return decorator


def listens_for(event_type: Type[BaseEvent], **kwargs):
    """
    Decorator to register a function as an event handler for a specific event type.

    The decorator introspects the decorated function's parameters, expecting type hints
    (often `typing.Annotated`) to declare dependencies like `WorldSession`,
    `WorldConfig`, specific `Resource` types, and the event object itself
    (e.g., `event: MySpecificEvent`).
    This information is stored in `SYSTEM_METADATA`.

    Args:
        event_type: The class of the event this system listens for (e.g., `AssetFileIngestionRequested`).
        **kwargs: Additional arbitrary keyword arguments for metadata.
    """
    if not (inspect.isclass(event_type) and issubclass(event_type, BaseEvent)):
        raise TypeError(f"Invalid event_type '{event_type}'. Must be a class that inherits from BaseEvent.")

    def decorator(func: Callable[..., Any]):
        # Decorator no longer adds to global EVENT_HANDLER_REGISTRY.
        # It primarily populates SYSTEM_METADATA.
        # Registration to a specific world's scheduler happens via world.register_system().
        param_info = _parse_system_params(func)
        SYSTEM_METADATA[func] = {
            "params": param_info,
            "is_async": inspect.iscoroutinefunction(func),
            "system_type": "event_handler",  # Mark as event-based
            "listens_for_event_type": event_type,
            **kwargs,
        }
        # Validate that the system actually has a parameter for this event_type
        has_event_param = any(p_info.get("event_type_hint") == event_type for p_info in param_info.values())
        if not has_event_param:
            # Try to find if any parameter has the event_type as its direct type_hint
            # This is covered by _parse_system_params if not Annotated.
            # If still not found, it's an issue.
            found_by_direct_type = any(
                p_info.get("type_hint") == event_type and p_info.get("identity") == "Event"
                for p_info in param_info.values()
            )
            if not found_by_direct_type:
                logger.warning(
                    f"System {func.__name__} registered for event {event_type.__name__} but does not "
                    f"seem to have a parameter matching this event type. Ensure one parameter is typed as "
                    f'`{event_type.__name__}` or `Annotated[{event_type.__name__}, "Event"]`.'
                )

        return func

    return decorator


logger = logging.getLogger(__name__)  # Module-level logger

# --- World Scheduler ---


class WorldScheduler:
    """
    Manages the execution of registered systems based on stages or events.

    The scheduler is responsible for:
    - Identifying which systems to run for a given stage or event from global registries.
    - Introspecting system function parameters to determine their dependencies.
    - Injecting required dependencies (e.g., database session from WorldContext,
      world configuration from WorldContext, resources from its own ResourceManager,
      lists of entities with specific markers, event objects).
    - Executing systems, supporting both asynchronous and synchronous functions.
    - Managing database session lifecycle (commit/rollback) per stage or per event dispatch group,
      using the session provided in WorldContext.
    """

    def __init__(self, resource_manager: ResourceManager):
        """
        Initializes the WorldScheduler.

        Args:
            resource_manager: An instance of ResourceManager specific to the World this
                              scheduler will operate within. This provides access to
                              world-specific shared resources.
        """
        self.resource_manager = resource_manager
        # WorldScheduler now has its own instance-level registries
        self.system_registry: Dict[SystemStage, List[Callable[..., Any]]] = defaultdict(list)
        self.event_handler_registry: Dict[Type[BaseEvent], List[Callable[..., Any]]] = defaultdict(list)
        # It still uses the global SYSTEM_METADATA for parameter info
        self.system_metadata = SYSTEM_METADATA
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")  # Scheduler instance logger

    def register_system_for_world(
        self,
        system_func: Callable[..., Any],
        stage: Optional[SystemStage] = None,
        event_type: Optional[Type[BaseEvent]] = None,
        **kwargs,  # These are metadata from the original decorator, can be stored if needed
    ):
        """
        Registers a system function (already decorated and in SYSTEM_METADATA)
        to this specific WorldScheduler instance for a stage or event.
        """
        if system_func not in self.system_metadata:
            # This implies the system wasn't decorated correctly with @system or @listens_for
            # Or _parse_system_params wasn't called for it during decoration.
            # For robust parsing, ensure @system or @listens_for also calls _parse_system_params
            # if it's not already guaranteed.
            # However, the current decorators do populate SYSTEM_METADATA.
            # One scenario: a raw function is passed that was never decorated.
            self.logger.warning(
                f"System function {system_func.__name__} is not in global SYSTEM_METADATA. "
                "Attempting to parse its parameters now. Ensure systems are decorated."
            )
            # Attempt to parse now, though ideally it's pre-parsed.
            # This might be redundant if decorators ensure metadata population.
            # Let's assume decorators handle metadata. If not, this is a fallback.
            if not _parse_system_params(system_func):  # Call to populate if missing
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
        """
        Helper to resolve dependencies and execute a single system (stage or event based).
        Dependencies are resolved using the provided world_context and the scheduler's resource_manager.
        """
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
                elif identity == "WorldContext":  # Task 3.2: Handle WorldContext injection
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
                        raise  # Re-raise to indicate failure to resolve dependencies
                elif identity == "MarkedEntityList":
                    marker_type = param_meta["marker_component_type"]
                    if not marker_type or not issubclass(marker_type, BaseComponent):
                        msg = (
                            f"System {system_func.__name__} has MarkedEntityList parameter '{param_name}' "
                            f"with invalid or missing marker component type in world '{world_context.world_name}'."
                        )
                        self.logger.error(msg)
                        raise ValueError(msg)

                    # Task 1.1: Optimized MarkedEntityList fetching using EXISTS
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
                    elif param_meta["event_type_hint"] is not None:
                        msg = (
                            f"System {system_func.__name__} parameter '{param_name}' in world '{world_context.world_name}' "
                            f"expects an event of type {param_meta['event_type_hint'].__name__} but none was provided."
                        )
                        self.logger.error(msg)
                        raise ValueError(msg)

        except Exception as e:
            self.logger.error(
                f"Error preparing dependencies for system {system_func.__name__} in world '{world_context.world_name}': {e}",
                exc_info=True,
            )
            return False

        self.logger.debug(
            f"Executing system: {system_func.__name__} in world '{world_context.world_name}' with args: {list(kwargs_to_inject.keys())}"
        )
        # try: # Task 5.1: Let exceptions propagate from here
        if metadata["is_async"]:
            await system_func(**kwargs_to_inject)
        else:
            # Run synchronous system in a thread pool executor to avoid blocking async event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: system_func(**kwargs_to_inject))
        # except Exception as e: # Task 5.1: Propagate instead of returning False
        # self.logger.error(f"Error executing system {system_func.__name__} in world '{world_context.world_name}': {e}", exc_info=True)
        # return False # Indicate system execution failure

        # Auto-remove marker component logic
        if metadata.get("system_type") == "stage_system":
            for param_name, param_meta in metadata["params"].items():
                if param_meta.get("identity") == "MarkedEntityList" and metadata.get("auto_remove_marker", True):
                    marker_type_to_remove = param_meta["marker_component_type"]
                    entities_processed = kwargs_to_inject.get(param_name, [])  # Should be populated if system ran
                    if entities_processed and marker_type_to_remove:
                        # Task 1.2: Optimized auto_remove_marker logic using bulk delete
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
                            # The flush here ensures that if the same system (or another in the same stage,
                            # if error handling changes) tries to re-evaluate this marker, it sees the change.
                            # It's consistent with the previous flush, relying on the overall stage commit/rollback.
                            world_context.session.flush()
        return True

    async def execute_stage(self, stage: SystemStage, world_context: WorldContext):
        """
        Executes all systems registered for a specific `SystemStage` within the given `world_context`.
        Handles session commit/rollback for the entire stage based on the success of its systems.
        The session in `world_context.session` is used and managed.
        """
        self.logger.info(f"Executing stage: {stage.name} for world: {world_context.world_name}")
        systems_to_run = self.system_registry.get(stage, [])
        if not systems_to_run:
            self.logger.info(f"No systems registered for stage {stage.name} in world {world_context.world_name}")
            return

        # all_systems_succeeded_in_stage = True # Task 5.1: Replaced with try/except block
        active_system_func_name = "None"
        try:
            for system_func in systems_to_run:
                active_system_func_name = system_func.__name__
                await self._resolve_and_execute_system(system_func, world_context)

            # If all systems executed without raising an exception
            try:
                world_context.session.commit()
                self.logger.info(f"Committed session for stage {stage.name} in world {world_context.world_name}")
            except Exception as commit_exc:
                self.logger.error(
                    f"Error committing session for stage {stage.name} in world {world_context.world_name}: {commit_exc}. Rolling back.",
                    exc_info=True,
                )
                world_context.session.rollback()
                # Raise StageExecutionError even for commit failure, as the stage didn't complete successfully.
                raise StageExecutionError(
                    message=f"Failed to commit stage {stage.name} in world {world_context.world_name}.",
                    stage_name=stage.name,
                    original_exception=commit_exc,
                ) from commit_exc
        except Exception as system_exc:  # Catch exceptions from _resolve_and_execute_system
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
        """
        Dispatches an event to all relevant event handlers within the given `world_context`.
        Handles session commit/rollback for the group of event handlers.
        The session in `world_context.session` is used and managed.
        """
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

            # If all handlers executed without raising an exception
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
        except Exception as handler_exc:  # Catch exceptions from _resolve_and_execute_system (via handler)
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
        """
        Executes all registered stages in their defined order using the provided initial_world_context.

        NOTE: This method is illustrative. In a real application, how sessions are managed
        across multiple stages (e.g., one session per stage vs. one session for all stages)
        would depend on transactional requirements. The current `execute_stage` method
        commits/rolls back the session passed via `world_context`. If `run_all_stages` is
        to manage a single session across all stages, `execute_stage` would need modification,
        or this method would need to handle the overarching transaction.

        For now, this method assumes that `initial_world_context.session` will be used
        and potentially committed/rolled back by each call to `execute_stage`.
        This means each stage effectively runs in its own transaction if `execute_stage`
        is called sequentially with the same session that gets committed/rolled back.
        A more robust `run_all_stages` would require careful session management strategy.
        The `World` object's `execute_stage` method handles creating a new session per call
        if one isn't provided, which is a safer default for isolated stage execution.
        """
        self.logger.info(f"Attempting to run all stages for world: {initial_world_context.world_name}")

        # This method is largely conceptual as session management across stages is complex.
        # The World object provides a more direct way to call execute_stage, managing sessions per call.
        # If this method were to be fully implemented, it would need to decide how to handle
        # the session from initial_world_context across multiple stage executions.
        # For example, does it pass the same session and expect execute_stage not to commit/close?
        # Or does it get a new session for each stage from a db_manager in initial_world_context?

        # For now, let's iterate through stages and call execute_stage, assuming
        # the session in initial_world_context will be used by each.
        # This implies that if a stage commits, subsequent stages operate on that committed state.
        # If a stage rolls back, subsequent stages operate on the rolled-back state.

        ordered_stages = sorted(list(SystemStage), key=lambda s: s.value if isinstance(s.value, int) else str(s.value))

        for stage in ordered_stages:
            self.logger.info(
                f"Running stage {stage.name} as part of run_all_stages for world {initial_world_context.world_name}."
            )
            # We use the initial_world_context, which carries the session.
            # The execute_stage method will then use this session and commit/rollback.
            await self.execute_stage(stage, initial_world_context)
            # If a stage fails and rolls back, the session is now in a rolled-back state.
            # Subsequent stages will operate on this. This might be desired or not.
            # If strict atomicity across all stages is needed, this approach is insufficient.

        self.logger.info(f"Finished running all stages for world: {initial_world_context.world_name}")


# Example Usage (conceptual, actual usage would be via a World instance):
# async def main_example_systems():
#     # 1. Setup: Create a World instance (which sets up its ResourceManager, WorldScheduler)
#     # This would typically happen at application startup or when a specific world is requested.
#     # from dam.core.world import create_and_register_world
#     # my_world = create_and_register_world("my_default_world_name") # Assuming config exists
#
#     # 2. Get a session for the world
#     # db_session = my_world.get_db_session()
#     # try:
#     #     # 3. Create WorldContext
#     #     world_ctx = WorldContext(session=db_session, world_name=my_world.name, world_config=my_world.config)
#     #
#     #     # 4. Execute a stage using the World's scheduler
#     #     # await my_world.scheduler.execute_stage(SystemStage.INGESTION, world_ctx)
#     #     # OR, more directly using the World's helper method:
#     #     await my_world.execute_stage(SystemStage.INGESTION, session=db_session) # World handles context creation
#     #
#     #     # 5. Dispatch an event
#     #     # from dam.core.events import SomeEventData, MyCustomEvent
#     #     # my_event_data = SomeEventData(info="example")
#     #     # an_event = MyCustomEvent(source="main_app", data=my_event_data)
#     #     # await my_world.dispatch_event(an_event, session=db_session)
#     #
#     # except Exception as e:
#     #     logger.error(f"An error occurred during system execution example: {e}", exc_info=True)
#     #     # db_session.rollback() # World.execute_stage/dispatch_event handles this if session is managed by them
#     # finally:
#     #     db_session.close() # Important to close sessions obtained directly
#
# # if __name__ == "__main__":
# #     asyncio.run(main_example_systems())
