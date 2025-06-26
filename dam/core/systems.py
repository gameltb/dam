import asyncio
import inspect
from collections import defaultdict
from typing import Annotated, Any, Callable, Dict, List, Type, get_args, get_origin, Optional # Added Optional

from dam.core.events import BaseEvent  # Import BaseEvent for type hinting
from dam.core.resources import ResourceManager, ResourceNotFoundError
from dam.core.stages import SystemStage
from dam.core.system_params import (
    WorldContext,
)
from dam.models import BaseComponent, Entity

# --- System Registration and Metadata ---

SYSTEM_REGISTRY: Dict[SystemStage, List[Callable[..., Any]]] = defaultdict(list)
"""
Global registry mapping SystemStage enums to a list of system functions.
Systems are added to this registry via the `@system` decorator.
"""

EVENT_HANDLER_REGISTRY: Dict[Type[BaseEvent], List[Callable[..., Any]]] = defaultdict(list)
"""
Global registry mapping Event types to a list of system functions that handle them.
Systems are added to this registry via the `@listens_for` decorator.
"""

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
        SYSTEM_REGISTRY[stage].append(func)
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
        EVENT_HANDLER_REGISTRY[event_type].append(func)
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
                 print(f"Warning: System {func.__name__} registered for event {event_type.__name__} "
                       f"but does not seem to have a parameter matching this event type. "
                       f"Ensure one parameter is typed as `{event_type.__name__}` or `Annotated[{event_type.__name__}, \"Event\"]`.")

        return func
    return decorator


# --- World Scheduler ---

class WorldScheduler:
    """
    Manages the execution of registered systems based on stages or events.

    The scheduler is responsible for:
    - Identifying which systems to run for a given stage or event.
    - Introspecting system function parameters to determine their dependencies.
    - Injecting required dependencies (e.g., database session, world configuration,
      resources from ResourceManager, lists of entities with specific markers, event objects).
    - Executing systems, supporting both asynchronous and synchronous functions.
    - Managing database session lifecycle (commit/rollback) per stage or per event dispatch group.
    """

    def __init__(self, resource_manager: ResourceManager):
        """
        Initializes the WorldScheduler.

        Args:
            resource_manager: An instance of ResourceManager to provide access to shared resources.
        """
        self.resource_manager = resource_manager
        self.system_registry = SYSTEM_REGISTRY
        self.event_handler_registry = EVENT_HANDLER_REGISTRY
        self.system_metadata = SYSTEM_METADATA

    async def _resolve_and_execute_system(
        self,
        system_func: Callable[..., Any],
        world_context: WorldContext,
        event_object: Optional[BaseEvent] = None # Pass event if it's an event handler
    ):
        """Helper to resolve dependencies and execute a single system (stage or event based)."""
        metadata = self.system_metadata.get(system_func)
        if not metadata:
            print(f"Warning: No metadata found for system {system_func.__name__}. Skipping.")
            return False # Indicate failure or skip

        kwargs_to_inject = {}
        try:
            for param_name, param_meta in metadata["params"].items():
                identity = param_meta["identity"]
                if identity == "WorldSession":
                    kwargs_to_inject[param_name] = world_context.session
                elif identity == "WorldName":
                    kwargs_to_inject[param_name] = world_context.world_name
                elif identity == "CurrentWorldConfig":
                    kwargs_to_inject[param_name] = world_context.world_config
                elif identity == "Resource":
                    try:
                        kwargs_to_inject[param_name] = self.resource_manager.get_resource(param_meta["type_hint"])
                    except ResourceNotFoundError as e:
                        raise ValueError(
                            f"System {system_func.__name__} requires resource {param_meta['type_hint'].__name__} which was not found: {e}"
                        )
                elif identity == "MarkedEntityList":
                    marker_type = param_meta["marker_component_type"]
                    if not marker_type or not issubclass(marker_type, BaseComponent):
                        raise ValueError(
                            f"System {system_func.__name__} has MarkedEntityList parameter '{param_name}' with invalid or missing marker component type."
                        )
                    from sqlalchemy import select as sql_select
                    stmt = sql_select(marker_type.entity_id).distinct()
                    entity_ids_with_marker = world_context.session.execute(stmt).scalars().all()
                    entities_to_process = []
                    if entity_ids_with_marker:
                        entities_to_process = (
                            world_context.session.query(Entity).filter(Entity.id.in_(entity_ids_with_marker)).all()
                        )
                    kwargs_to_inject[param_name] = entities_to_process
                    print(
                        f"System {system_func.__name__} gets {len(entities_to_process)} entities for marker {marker_type.__name__}"
                    )
                elif identity == "Event":
                    # Check if the event_object provided matches the parameter's expected event type
                    expected_event_type = param_meta["event_type_hint"]
                    if event_object and isinstance(event_object, expected_event_type):
                        kwargs_to_inject[param_name] = event_object
                    elif event_object: # Mismatch
                        # This should ideally be caught by the dispatch logic that only calls compatible handlers
                        print(f"Warning: System {system_func.__name__} expected event type {expected_event_type} "
                              f"but received {type(event_object)}. Skipping injection for this param.")
                    else: # No event object provided (e.g. if called from stage execution)
                        # This indicates a system designed as an event handler was called in a non-event context
                        # or an event handler is missing its event type hint properly.
                        # For now, we allow it, but it might lead to errors if the system *requires* the event.
                        # Better: raise error if param_meta["event_type_hint"] is not None and event_object is None
                        if param_meta["event_type_hint"] is not None:
                             raise ValueError(f"System {system_func.__name__} parameter '{param_name}' expects an event of type "
                                              f"{param_meta['event_type_hint'].__name__} but none was provided (event_object is None).")


        except Exception as e:
            print(f"Error preparing dependencies for system {system_func.__name__}: {e}")
            return False # Indicate failure

        print(
            f"Executing system: {system_func.__name__} with args: {list(kwargs_to_inject.keys())}"
        )
        if metadata["is_async"]:
            await system_func(**kwargs_to_inject)
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: system_func(**kwargs_to_inject))

        # Auto-remove marker component logic (applies if it was a stage system with MarkedEntityList)
        if metadata.get("system_type") == "stage_system": # Only for stage systems
            for param_name, param_meta in metadata["params"].items(): # Re-iterate to find MarkedEntityList
                 if param_meta.get("identity") == "MarkedEntityList" and metadata.get("auto_remove_marker", True):
                    marker_type_to_remove = param_meta["marker_component_type"]
                    entities_processed = kwargs_to_inject.get(param_name, [])
                    if entities_processed and marker_type_to_remove:
                        from dam.services import ecs_service  # Local import
                        print(
                            f"Scheduler: Removing {marker_type_to_remove.__name__} from {len(entities_processed)} entities after system {system_func.__name__}."
                        )
                        for entity_obj in entities_processed:
                            comp_to_remove = ecs_service.get_component(
                                world_context.session, entity_obj.id, marker_type_to_remove
                            )
                            if comp_to_remove:
                                ecs_service.remove_component(
                                    world_context.session, comp_to_remove, flush=False
                                )
                        world_context.session.flush() # Flush removals for this system's markers
        return True # Indicate success

    async def execute_stage(self, stage: SystemStage, world_context: WorldContext):
        """
        Executes all systems registered for a specific `SystemStage`.
        Handles session commit/rollback for the entire stage.
        """
        print(f"Executing stage: {stage.name} for world: {world_context.world_name}")
        systems_to_run = self.system_registry.get(stage, [])
        if not systems_to_run:
            print(f"No systems registered for stage {stage.name}")
            return

        all_systems_succeeded = True
        for system_func in systems_to_run:
            success = await self._resolve_and_execute_system(system_func, world_context)
            if not success:
                all_systems_succeeded = False
                # Decide if stage execution should stop on first system error, or try all.
                # For now, let's try all, then rollback if any failed.
                # Or, more strictly: stop and rollback on first error.
                print(f"System {system_func.__name__} failed in stage {stage.name}. Stage will be rolled back.")
                break # Stop on first error

        if all_systems_succeeded:
            try:
                world_context.session.commit()
                print(f"Committed session for stage {stage.name} in world {world_context.world_name}")
            except Exception as e:
                print(
                    f"Error committing session for stage {stage.name} in world {world_context.world_name}: {e}. Rolling back."
                )
                world_context.session.rollback()
                raise # Re-raise commit error
        else:
            print(f"One or more systems failed in stage {stage.name}. Rolling back session for world {world_context.world_name}.")
            world_context.session.rollback()
            # Optionally raise an exception to signal stage failure to caller
            # raise StageExecutionError(f"Stage {stage.name} failed due to system errors.")

    async def dispatch_event(self, event: BaseEvent, world_context: WorldContext):
        """
        Dispatches an event to all registered event handlers for its type.
        Handles session commit/rollback for the group of event handlers.
        """
        event_type = type(event)
        print(f"Dispatching event: {event_type.__name__} for world: {world_context.world_name}")

        handlers_to_run = self.event_handler_registry.get(event_type, [])
        if not handlers_to_run:
            print(f"No event handlers registered for event type {event_type.__name__}")
            return

        all_handlers_succeeded = True
        for handler_func in handlers_to_run:
            # Ensure the handler is actually designed for this specific event type,
            # though registration should ensure this.
            # The _resolve_and_execute_system will match param type with event object type.
            success = await self._resolve_and_execute_system(handler_func, world_context, event_object=event)
            if not success:
                all_handlers_succeeded = False
                print(f"Event handler {handler_func.__name__} failed for event {event_type.__name__}. Event processing will be rolled back.")
                break # Stop on first error

        if all_handlers_succeeded:
            try:
                world_context.session.commit()
                print(f"Committed session after handling event {event_type.__name__} in world {world_context.world_name}")
            except Exception as e:
                print(
                    f"Error committing session after event {event_type.__name__} in world {world_context.world_name}: {e}. Rolling back."
                )
                world_context.session.rollback()
                raise # Re-raise commit error
        else:
            print(f"One or more handlers failed for event {event_type.__name__}. Rolling back session for world {world_context.world_name}.")
            world_context.session.rollback()
            # Optionally raise an exception
            # raise EventHandlingError(f"Event {event_type.__name__} handling failed.")


    async def run_all_stages(self, initial_world_context: WorldContext):
        """
        Executes all registered stages in their defined order.
        NOTE: This is a simplified sequential execution. A real app might have more complex logic
        for when and how stages are run. Also, WorldContext might need to be refreshed (e.g. new session)
        per stage or per group of stages depending on transaction semantics.
        For now, we'll create a new session for each stage from the initial world_context's db_manager.
        This implies db_manager needs to be part of WorldContext or accessible.
        Let's assume initial_world_context has a way to get a new session.
        """
        # This needs a db_manager to create new sessions per stage.
        # Let's assume WorldContext can provide a session factory or similar.
        # For now, this is a placeholder for more robust multi-stage execution.
        # The current execute_stage commits and closes session, so each stage needs a new one.

        # This part needs access to the db_manager to create new sessions for each stage.
        # For now, this method is illustrative and not fully functional without that.
        print("Running all stages - this part is illustrative and needs db_manager access for new sessions per stage.")

        # A proper implementation would iterate through SystemStage enum values in order.
        # ordered_stages = sorted(list(SystemStage), key=lambda s: s.value) # If Enum has implicit order
        # For now, using the order they appear in the Enum definition (Python 3.6+ behavior)
        # ordered_stages = list(SystemStage) # Unused variable

        # This is a simplified loop. A real app would need a way to get a fresh session for each stage
        # if the previous stage's session was closed.
        # For now, this shows the intent but won't work correctly if sessions are closed by execute_stage.
        # The execute_stage should probably NOT close the session if part of a larger run_all_stages flow,
        # or run_all_stages must manage session lifecycle across stages.

        # Let's assume for now that `execute_stage` is called externally for each stage,
        # and the caller manages the session for that stage.
        # So, `run_all_stages` is more of a conceptual guide here.
        pass


# Example Usage (conceptual, would be in main application logic)
# async def main():
#     resource_mgr = ResourceManager()
#     resource_mgr.add_resource(FileOperationsResource())
#
#     scheduler = WorldScheduler(resource_mgr)
#
#     # Assume db_manager is available and configured
#     from dam.core.database import db_manager
#     world_name = "my_world"
#
#     # For each stage execution, a new session and context would be created
#     async with db_manager.get_db_session_async(world_name) as session: # Hypothetical async session
#         world_ctx = WorldContext(session, world_name, db_manager.settings.get_world_config(world_name))
#         await scheduler.execute_stage(SystemStage.METADATA_EXTRACTION, world_ctx)

#     # Or if stages manage their own sessions via context:
#     # await scheduler.execute_stage(SystemStage.METADATA_EXTRACTION, world_name)
#     # where execute_stage internally gets session for that world.
#     # The current execute_stage expects session in WorldContext.
