import asyncio
import inspect
from collections import defaultdict
from typing import Annotated, Any, Callable, Dict, List, get_args, get_origin  # Added Annotated

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

SYSTEM_METADATA: Dict[Callable[..., Any], Dict[str, Any]] = {}
"""
Global dictionary storing metadata for each registered system function.
Keys are the system functions themselves.
Values are dictionaries containing parsed parameter information (name, type, identity),
async status, and any other kwargs passed to the decorator.
Example:
{
    my_system_func: {
        "params": {
            "param1_name": {"name": "param1", "type_hint": Session, "identity": "WorldSession", ...},
            "param2_name": {"name": "param2", "type_hint": MyResource, "identity": "Resource", ...}
        },
        "is_async": True,
        "custom_decorator_arg": "value"
    }
}
"""


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
        """
        Inner decorator that performs the actual registration and metadata parsing.
        """
        SYSTEM_REGISTRY[stage].append(func)

        sig = inspect.signature(func)
        param_info = {}
        for name, param in sig.parameters.items():
            param_type = param.annotation
            identity_type = None
            actual_type = param_type
            marker_component_type = None  # For MarkedEntityList

            if get_origin(param_type) is Annotated:
                annotations = get_args(param_type)  # param.annotation holds the full Annotated type
                actual_type = annotations[0]  # The actual type, e.g., List[Entity]
                identity_type_str = None
                marker_type_from_annotated = None

                for ann_arg in annotations[1:]:  # Iterate through metadata parts of Annotated
                    if isinstance(ann_arg, str):
                        identity_type_str = ann_arg
                    elif inspect.isclass(ann_arg) and issubclass(ann_arg, BaseComponent):  # Check if it's a class
                        marker_type_from_annotated = ann_arg

                identity_type = identity_type_str  # Assign the found string identity

                if identity_type_str == "MarkedEntityList":
                    marker_component_type = marker_type_from_annotated  # Assign found marker type
                # Ensure other identities don't incorrectly get a marker_component_type
                elif marker_component_type is not None and identity_type_str != "MarkedEntityList":
                    # This case should ideally not happen if annotations are used correctly
                    # but good for robustness if other Annotated uses involve component types.
                    # For now, only MarkedEntityList uses the third Annotated arg for a component type.
                    pass

            param_info[name] = {
                "name": name,
                "type_hint": actual_type,
                "identity": identity_type,  # "WorldSession", "Resource", "MarkedEntityList", etc.
                "marker_component_type": marker_component_type,  # Store the M in MarkedEntityList[M]
                "is_annotated": get_origin(param_type) is Annotated,
                "original_annotation": param.annotation,
            }
        SYSTEM_METADATA[func] = {"params": param_info, "is_async": inspect.iscoroutinefunction(func), **kwargs}
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
      resources from ResourceManager, lists of entities with specific markers).
    - Executing systems, supporting both asynchronous and synchronous functions
      (synchronous functions are run in a thread pool to avoid blocking the event loop).
    - Managing database session lifecycle (commit/rollback) per stage.
    """

    def __init__(self, resource_manager: ResourceManager):
        """
        Initializes the WorldScheduler.

        Args:
            resource_manager: An instance of ResourceManager to provide access to shared resources.
        """
        self.resource_manager = resource_manager
        self.system_registry = SYSTEM_REGISTRY  # Uses the global registry
        self.system_metadata = SYSTEM_METADATA  # Uses the global metadata store

    async def execute_stage(self, stage: SystemStage, world_context: WorldContext):
        """
        Executes all systems registered for a specific `SystemStage`.

        For each system, it resolves and injects declared dependencies such as:
        - `WorldSession`: The active SQLAlchemy session for the current world.
        - `WorldName`: The name of the current world.
        - `CurrentWorldConfig`: The configuration object for the current world.
        - `Resource[SomeResourceType]`: A specific resource from the `ResourceManager`.
        - `MarkedEntityList[SomeMarkerComponent]`: A list of `Entity` objects that currently
          have the specified `SomeMarkerComponent` attached.

        The method handles session commits after all systems in a stage have run,
        or rolls back in case of errors during system execution or commit.

        Args:
            stage: The `SystemStage` to execute.
            world_context: A `WorldContext` object providing the database session,
                           world name, and world configuration for this execution.

        Raises:
            ValueError: If a system declares a dependency that cannot be resolved (e.g.,
                        missing resource, invalid marker component type).
        """
        # Changed to print, consider using structured logging if this becomes more complex
        print(f"Executing stage: {stage.name} for world: {world_context.world_name}")

        systems_to_run = self.system_registry.get(stage, [])
        if not systems_to_run:
            print(f"No systems registered for stage {stage.name}")  # Changed to print
            return

        for system_func in systems_to_run:
            metadata = self.system_metadata.get(system_func)
            if not metadata:
                print(f"Warning: No metadata found for system {system_func.__name__}. Skipping.")  # Changed to print
                continue

            kwargs_to_inject = {}
            try:
                for param_name, param_meta in metadata["params"].items():
                    if param_meta["identity"] == "WorldSession":
                        kwargs_to_inject[param_name] = world_context.session
                    elif param_meta["identity"] == "WorldName":
                        kwargs_to_inject[param_name] = world_context.world_name
                    elif param_meta["identity"] == "CurrentWorldConfig":
                        kwargs_to_inject[param_name] = world_context.world_config
                    elif param_meta["identity"] == "Resource":
                        try:
                            kwargs_to_inject[param_name] = self.resource_manager.get_resource(param_meta["type_hint"])
                        except ResourceNotFoundError as e:
                            raise ValueError(
                                f"System {system_func.__name__} requires resource {param_meta['type_hint'].__name__} which was not found: {e}"
                            )
                    elif param_meta["identity"] == "MarkedEntityList":
                        marker_type = param_meta["marker_component_type"]
                        if not marker_type or not issubclass(marker_type, BaseComponent):
                            raise ValueError(
                                f"System {system_func.__name__} has MarkedEntityList parameter '{param_name}' with invalid or missing marker component type."
                            )

                        # This import needs to be here or ecs_service needs to be a resource
                        from dam.services import ecs_service

                        # Query entities with the marker.
                        # This is a simplified query. A real implementation might be more optimized
                        # or allow querying for components directly.
                        # all_entities_with_marker_component = [] # Unused variable
                        # This is inefficient - ideally query entities that have this component type.
                        # For now, let's assume ecs_service can provide this.
                        # This is a placeholder for a proper query.
                        # For example: entities = ecs_service.query_entities_with_component(world_context.session, marker_type)

                        # Simplified placeholder: get all entities and filter. VERY INEFFICIENT.
                        # In a real scenario, you'd have a way to query entities *having* a certain component.
                        # For example, `SELECT entity_id FROM component_marker_needs_metadata_extraction;`
                        # then fetch those entities.
                        # For now, let's assume it's provided or the system does it internally.
                        # This part needs a proper ECS query mechanism.
                        # Let's assume for now the system receives the session and queries itself,
                        # or this MarkedEntityList implies a pre-fetch.
                        # For this PoC, we'll pass an empty list and the system needs to handle it.

                        # A better approach for MarkedEntityList:
                        # The scheduler queries for entities that have the marker_type component.
                        from sqlalchemy import select as sql_select  # Import select

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

                    # Add more parameter identity handlers here (e.g., ComponentQuery, EventHandlerFor)

                    # If a parameter is not handled by specific identity, it's an error for non-on-demand systems.
                    # On-demand systems would expect these to be passed by the caller.
                    # This logic needs to be more robust for on-demand vs. staged systems.

            except Exception as e:
                print(f"Error preparing dependencies for system {system_func.__name__}: {e}")  # Changed to print
                continue  # Skip this system

            print(
                f"Executing system: {system_func.__name__} with args: {list(kwargs_to_inject.keys())}"
            )  # Changed to print
            if metadata["is_async"]:
                await system_func(**kwargs_to_inject)
            else:
                # For synchronous systems, run in a thread to avoid blocking asyncio loop
                # Note: SQLAlchemy sessions are generally not thread-safe.
                # Synchronous systems needing a session would typically create their own
                # or use a session from a thread-local scope if the DB manager supports that.
                # For simplicity now, if a sync system needs WorldSession, this might be problematic.
                # A better approach is to encourage async systems or have them manage their own sync session.
                # For now, if it's sync and requests WorldSession, it will get the main thread's session.

                # This is a simplified call. Proper handling of sync systems in an async scheduler
                # requires careful thought about blocking and resource sharing (like sessions).
                # Using to_thread is a good general approach for I/O-bound sync code.

                # We need to ensure that the arguments passed to to_thread are safe.
                # The session object might not be.
                # A common pattern is for the sync function to create its own session if needed.
                # For now, this is a simplification.
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: system_func(**kwargs_to_inject))

            # After system execution, for MarkedEntityList, we might want to remove the marker.
            # This could be a responsibility of the system itself, or the scheduler.
            # If scheduler does it:
            if param_meta.get("identity") == "MarkedEntityList" and param_meta.get(
                "auto_remove_marker", True
            ):  # Add auto_remove_marker option
                marker_type_to_remove = param_meta["marker_component_type"]
                entities_processed = kwargs_to_inject.get(param_name, [])
                if entities_processed and marker_type_to_remove:
                    from dam.services import ecs_service  # Local import

                    print(
                        f"Scheduler: Removing {marker_type_to_remove.__name__} from {len(entities_processed)} entities."
                    )
                    for entity_obj in entities_processed:
                        comp_to_remove = ecs_service.get_component(
                            world_context.session, entity_obj.id, marker_type_to_remove
                        )
                        if comp_to_remove:
                            ecs_service.remove_component(
                                world_context.session, comp_to_remove, flush=False
                            )  # Batch flush later
                    world_context.session.flush()  # Flush removals for this system

        print(f"Finished stage: {stage.name} for world: {world_context.world_name}")  # Changed to print
        try:
            # Systems might have flushed changes. The stage itself will commit.
            # This commit makes changes from this stage visible to subsequent stages
            # if they use a new session, or to the caller if using the same session.
            world_context.session.commit()
            print(f"Committed session for stage {stage.name} in world {world_context.world_name}")
        except Exception as e:
            print(
                f"Error committing session for stage {stage.name} in world {world_context.world_name}: {e}. Rolling back."
            )
            world_context.session.rollback()
            raise  # Re-raise the exception so the caller knows the stage failed
        # Session closing is now responsibility of the caller that created/provided the session in WorldContext.

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
