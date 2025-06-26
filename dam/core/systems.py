import asyncio
import inspect
import logging # Added logging
from collections import defaultdict
from typing import Annotated, Any, Callable, Dict, List, Type, get_args, get_origin, Optional

from dam.core.events import BaseEvent
from dam.core.resources import ResourceManager, ResourceNotFoundError
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


def _parse_system_params(func: Callable[..., Any]) -> Dict[str, Any]:
    """Helper function to parse parameters of a system function."""
    sig = inspect.signature(func)
    param_info = {}
    for name, param in sig.parameters.items():
        param_type = param.annotation
        identity_type = None
        actual_type = param_type
        marker_component_type = None  # For MarkedEntityList
        event_type_hint = None # For Event parameters

        if get_origin(param_type) is Annotated:
            annotations = get_args(param_type)
            actual_type = annotations[0]
            identity_type_str = None
            marker_type_from_annotated = None

            for ann_arg in annotations[1:]:
                if isinstance(ann_arg, str):
                    identity_type_str = ann_arg
                elif inspect.isclass(ann_arg) and issubclass(ann_arg, BaseComponent):
                    marker_type_from_annotated = ann_arg
                elif inspect.isclass(ann_arg) and issubclass(ann_arg, BaseEvent): # Check for Event annotation
                    # This assumes an Annotated Event type like Annotated[MyEvent, "Event"]
                    # Or the identity "Event" is used to mark it, and actual_type is the event.
                    pass # Handled by identity_type_str == "Event"

            identity_type = identity_type_str

            if identity_type_str == "MarkedEntityList":
                marker_component_type = marker_type_from_annotated
            elif identity_type_str == "Event":
                # The 'actual_type' (e.g., MyCustomEvent) is the important part for event handlers
                # It will be used to match the dispatched event type.
                event_type_hint = actual_type


        # If not annotated, but type is a subclass of BaseEvent, assume it's an event parameter
        # This allows systems to declare `my_event: MySpecificEvent` directly.
        elif inspect.isclass(actual_type) and issubclass(actual_type, BaseEvent):
            identity_type = "Event"
            event_type_hint = actual_type

        param_info[name] = {
            "name": name,
            "type_hint": actual_type,
            "identity": identity_type,
            "marker_component_type": marker_component_type,
            "event_type_hint": event_type_hint, # Store specific event type if this param is an event
            "is_annotated": get_origin(param_type) is Annotated,
            "original_annotation": param.annotation,
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
            "system_type": "stage_system", # Mark as stage-based
            **kwargs
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
            "system_type": "event_handler", # Mark as event-based
            "listens_for_event_type": event_type,
            **kwargs
        }
        # Validate that the system actually has a parameter for this event_type
        has_event_param = any(
            p_info.get("event_type_hint") == event_type for p_info in param_info.values()
        )
        if not has_event_param:
            # Try to find if any parameter has the event_type as its direct type_hint
            # This is covered by _parse_system_params if not Annotated.
            # If still not found, it's an issue.
            found_by_direct_type = any(
                p_info.get("type_hint") == event_type and p_info.get("identity") == "Event"
                for p_info in param_info.values()
            )
            if not found_by_direct_type:
                 logger.warning(f"System {func.__name__} registered for event {event_type.__name__} "
                       f"but does not seem to have a parameter matching this event type. "
                       f"Ensure one parameter is typed as `{event_type.__name__}` or `Annotated[{event_type.__name__}, \"Event\"]`.")

        return func
    return decorator

logger = logging.getLogger(__name__) # Module-level logger

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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}") # Scheduler instance logger

    def register_system_for_world(
        self,
        system_func: Callable[..., Any],
        stage: Optional[SystemStage] = None,
        event_type: Optional[Type[BaseEvent]] = None,
        **kwargs # These are metadata from the original decorator, can be stored if needed
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
            if not _parse_system_params(system_func): # Call to populate if missing
                 self.logger.error(f"Failed to parse parameters for undecorated system {system_func.__name__}. Cannot register.")
                 return


        if stage:
            self.system_registry[stage].append(system_func)
            self.logger.info(f"System {system_func.__name__} registered for stage {stage.name} in this scheduler.")
        elif event_type:
            self.event_handler_registry[event_type].append(system_func)
            self.logger.info(f"System {system_func.__name__} registered for event {event_type.__name__} in this scheduler.")
        else:
            self.logger.error(f"System {system_func.__name__} must be registered with either a stage or an event_type.")


    async def _resolve_and_execute_system(
        self,
        system_func: Callable[..., Any],
        world_context: WorldContext,
        event_object: Optional[BaseEvent] = None
    ):
        """
        Helper to resolve dependencies and execute a single system (stage or event based).
        Dependencies are resolved using the provided world_context and the scheduler's resource_manager.
        """
        metadata = self.system_metadata.get(system_func)
        if not metadata:
            self.logger.warning(f"No metadata found for system {system_func.__name__} in world '{world_context.world_name}'. Skipping.")
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
                elif identity == "Resource":
                    try:
                        kwargs_to_inject[param_name] = self.resource_manager.get_resource(param_type_hint)
                    except ResourceNotFoundError as e:
                        self.logger.error(
                            f"System {system_func.__name__} in world '{world_context.world_name}' requires resource "
                            f"{param_type_hint.__name__} which was not found: {e}", exc_info=True
                        )
                        raise # Re-raise to indicate failure to resolve dependencies
                elif identity == "MarkedEntityList":
                    marker_type = param_meta["marker_component_type"]
                    if not marker_type or not issubclass(marker_type, BaseComponent):
                        msg = (f"System {system_func.__name__} has MarkedEntityList parameter '{param_name}' "
                               f"with invalid or missing marker component type in world '{world_context.world_name}'.")
                        self.logger.error(msg)
                        raise ValueError(msg)

                    from sqlalchemy import select as sql_select # Keep local import for SQLAlchemy specifics
                    stmt = sql_select(marker_type.entity_id).distinct()
                    entity_ids_with_marker = world_context.session.execute(stmt).scalars().all()
                    entities_to_process = []
                    if entity_ids_with_marker:
                        entities_to_process = (
                            world_context.session.query(Entity).filter(Entity.id.in_(entity_ids_with_marker)).all()
                        )
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
                        msg = (f"System {system_func.__name__} parameter '{param_name}' in world '{world_context.world_name}' "
                               f"expects an event of type {param_meta['event_type_hint'].__name__} but none was provided.")
                        self.logger.error(msg)
                        raise ValueError(msg)

        except Exception as e:
            self.logger.error(
                f"Error preparing dependencies for system {system_func.__name__} in world '{world_context.world_name}': {e}",
                exc_info=True
            )
            return False

        self.logger.debug(
            f"Executing system: {system_func.__name__} in world '{world_context.world_name}' with args: {list(kwargs_to_inject.keys())}"
        )
        try:
            if metadata["is_async"]:
                await system_func(**kwargs_to_inject)
            else:
                # Run synchronous system in a thread pool executor to avoid blocking async event loop
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: system_func(**kwargs_to_inject))
        except Exception as e:
            self.logger.error(f"Error executing system {system_func.__name__} in world '{world_context.world_name}': {e}", exc_info=True)
            return False # Indicate system execution failure


        # Auto-remove marker component logic
        if metadata.get("system_type") == "stage_system":
            for param_name, param_meta in metadata["params"].items():
                 if param_meta.get("identity") == "MarkedEntityList" and metadata.get("auto_remove_marker", True):
                    marker_type_to_remove = param_meta["marker_component_type"]
                    entities_processed = kwargs_to_inject.get(param_name, []) # Should be populated if system ran
                    if entities_processed and marker_type_to_remove:
                        # Local import to avoid circular dependencies if ecs_service imports systems
                        from dam.services import ecs_service
                        self.logger.debug(
                            f"Scheduler for world '{world_context.world_name}': Removing {marker_type_to_remove.__name__} from "
                            f"{len(entities_processed)} entities after system {system_func.__name__}."
                        )
                        for entity_obj in entities_processed:
                            comp_to_remove = ecs_service.get_component(
                                world_context.session, entity_obj.id, marker_type_to_remove
                            )
                            if comp_to_remove:
                                ecs_service.remove_component(
                                    world_context.session, comp_to_remove, flush=False # Batch flush
                                )
                        world_context.session.flush() # Flush all marker removals for this system
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

        all_systems_succeeded_in_stage = True
        for system_func in systems_to_run:
            system_success = await self._resolve_and_execute_system(system_func, world_context)
            if not system_success:
                all_systems_succeeded_in_stage = False
                self.logger.error(
                    f"System {system_func.__name__} failed in stage {stage.name} for world {world_context.world_name}. "
                    "Stage execution will be rolled back."
                )
                break # Stop on first system error within the stage

        if all_systems_succeeded_in_stage:
            try:
                world_context.session.commit()
                self.logger.info(f"Committed session for stage {stage.name} in world {world_context.world_name}")
            except Exception as e:
                self.logger.error(
                    f"Error committing session for stage {stage.name} in world {world_context.world_name}: {e}. Rolling back.",
                    exc_info=True
                )
                world_context.session.rollback()
                # Consider re-raising to signal failure to the caller (e.g., the World object)
                # raise StageExecutionError(...) from e
        else:
            self.logger.warning(
                f"One or more systems failed in stage {stage.name}. Rolling back session for world {world_context.world_name}."
            )
            world_context.session.rollback()
            # Optionally raise an exception to signal stage failure
            # raise StageExecutionError(f"Stage {stage.name} failed due to system errors in world {world_context.world_name}.")


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
            self.logger.info(f"No event handlers registered for event type {event_type.__name__} in world {world_context.world_name}")
            return

        all_handlers_succeeded = True
        for handler_func in handlers_to_run:
            handler_success = await self._resolve_and_execute_system(handler_func, world_context, event_object=event)
            if not handler_success:
                all_handlers_succeeded = False
                self.logger.error(
                    f"Event handler {handler_func.__name__} failed for event {event_type.__name__} "
                    f"in world {world_context.world_name}. Event processing group will be rolled back."
                )
                break # Stop on first handler error for this event

        if all_handlers_succeeded:
            try:
                world_context.session.commit()
                self.logger.info(f"Committed session after handling event {event_type.__name__} in world {world_context.world_name}")
            except Exception as e:
                self.logger.error(
                    f"Error committing session after event {event_type.__name__} in world {world_context.world_name}: {e}. Rolling back.",
                    exc_info=True
                )
                world_context.session.rollback()
                # raise EventHandlingError(...) from e
        else:
            self.logger.warning(
                f"One or more handlers failed for event {event_type.__name__}. Rolling back session for world {world_context.world_name}."
            )
            world_context.session.rollback()
            # raise EventHandlingError(f"Event {event_type.__name__} handling failed in world {world_context.world_name}.")


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
            self.logger.info(f"Running stage {stage.name} as part of run_all_stages for world {initial_world_context.world_name}.")
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
