from typing import Annotated, List, Type, TypeVar

from sqlalchemy.orm import Session

from dam.core.config import WorldConfig  # Assuming WorldConfig is in dam.core.config
from dam.models import BaseComponent, Entity

# --- Parameter Identity Annotations ---

# Represents the current SQLAlchemy session for the active world.
# Injected by the WorldScheduler.
WorldSession = Annotated[Session, "WorldSession"]

# Represents the name of the current active world.
# Injected by the WorldScheduler.
WorldName = Annotated[str, "WorldName"]

# Represents the configuration object for the current active world.
# Injected by the WorldScheduler.
CurrentWorldConfig = Annotated[WorldConfig, "CurrentWorldConfig"]

# Generic type for resources managed by the ResourceManager.
# Systems can request a resource by its type.
# Usage: my_service: Resource[MyServiceClass]
R = TypeVar("R")
Resource = Annotated[R, "Resource"]

# Represents a list of entities that have a specific marker component.
# When a system parameter is type-hinted with this, the WorldScheduler will:
# 1. Identify the `SomeMarkerComponent` from the annotation.
# 2. Query the database for all `Entity` instances that currently have `SomeMarkerComponent` attached.
# 3. Inject this list of entities into the system parameter.
#
# Usage in a system function:
#   from dam.core.components_markers import NeedsProcessingMarker
#   def my_system(
#       entities: Annotated[List[Entity], "MarkedEntityList", NeedsProcessingMarker]
#   ):
#       for entity in entities:
#           # process entity
#
# The marker component type (e.g., `NeedsProcessingMarker`) must be a class that
# inherits from `BaseComponent`.
M = TypeVar("M", bound=Type[BaseComponent])  # Ensures M is a type of a component
MarkedEntityList = Annotated[List[Entity], "MarkedEntityList", M]
"""
Type alias for a list of Entities that are marked with a specific component.
The third argument in `Annotated` (M) specifies the marker component class.
"""

# Placeholder for a more advanced component query system (Bevy-like Query).
# This would involve a custom Query type and more complex parsing by the scheduler.
# For now, systems can fetch components manually using ecs_service or MarkedEntityList.
# Q_Components = TypeVar('Q_Components', bound=Tuple[Type[BaseComponent], ...])
# ComponentQuery = Annotated[List[Tuple[Entity, ...]], "ComponentQuery", Q_Components]

# Placeholder for event handler parameters.
# If systems are triggered by specific events.
# E = TypeVar('E') # Event type
# EventHandlerFor = Annotated[E, "EventHandlerFor"]


# --- World Context Object (used by WorldScheduler) ---


class WorldContext:
    """
    A data class that bundles together world-specific contextual information
    required by the `WorldScheduler` to execute a stage or process events.

    This object is typically constructed by the application logic (e.g., a CLI command handler)
    that initiates a scheduler operation for a specific world. It is then passed to
    methods like `WorldScheduler.execute_stage()`.

    Attributes:
        session: The active SQLAlchemy `Session` for the world.
        world_name: The string name of the world being processed.
        world_config: The `WorldConfig` object containing settings for this world.
    """

    def __init__(self, session: Session, world_name: str, world_config: WorldConfig):
        self.session = session
        self.world_name = world_name
        self.world_config = world_config

    # Note: While WorldContext holds session, world_name, and world_config,
    # systems should declare these as dependencies via Annotated types (WorldSession,
    # WorldName, CurrentWorldConfig) rather than directly accessing WorldContext fields.
    # This promotes loose coupling and allows the scheduler to manage how these are provided.
