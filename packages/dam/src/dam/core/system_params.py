from typing import Annotated, List, Type, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession  # Import AsyncSession

# Corrected imports for BaseComponent and Entity
from dam.models.core.base_component import BaseComponent
from dam.models.core.entity import Entity

# --- Parameter Identity Annotations ---

# Represents the current SQLAlchemy session for the active world.
# Injected by the WorldScheduler.
WorldSession = Annotated[AsyncSession, "WorldSession"]  # Changed to AsyncSession

# WorldName and CurrentWorldConfig are no longer special DI identities.
# Systems should inject WorldConfig by its type (it's a resource)
# and access world_config.name if the name is needed.
# WorldName = Annotated[str, "WorldName"] # Removed
# CurrentWorldConfig = Annotated[WorldConfig, "CurrentWorldConfig"] # Removed

# Generic type for resources managed by the ResourceManager.
# Systems can request a resource by its type.
# Usage: my_resource: Resource[MyResourceClass]
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
# For now, systems can fetch components manually using ecs_functions or MarkedEntityList.
# Q_Components = TypeVar('Q_Components', bound=Tuple[Type[BaseComponent], ...])
# ComponentQuery = Annotated[List[Tuple[Entity, ...]], "ComponentQuery", Q_Components]

# Placeholder for event handler parameters.
# If systems are triggered by specific events.
# E = TypeVar('E') # Event type
# EventHandlerFor = Annotated[E, "EventHandlerFor"]
