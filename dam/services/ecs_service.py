from typing import Type, TypeVar, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select

from dam.models import Entity, BaseComponent

# Define a generic type variable for component types
T = TypeVar('T', bound=BaseComponent)

# Placeholder for future ECS service functions.

def get_entity(session: Session, entity_id: int) -> Optional[Entity]:
    """Retrieves an entity by its ID."""
    return session.get(Entity, entity_id)

def add_component_to_entity(
    session: Session,
    entity_id: int,
    component_instance: T
) -> T:
    """
    Adds a component instance to a specified entity.

    Args:
        session: The SQLAlchemy session.
        entity_id: The ID of the entity to add the component to.
        component_instance: An instance of a class inheriting from BaseComponent.
                            The instance should have its specific fields already populated.

    Returns:
        The added component instance, now associated with the entity and session.

    Raises:
        ValueError: If the entity with the given entity_id is not found.
    """
    entity = get_entity(session, entity_id)
    if not entity:
        raise ValueError(f"Entity with ID {entity_id} not found.")

    # Associate component with the entity
    # These fields are kw_only in BaseComponent's effective __init__
    # For direct instantiation, both entity and entity_id are now expected by tests
    # However, for adding an *existing* component instance, we just set them.
    component_instance.entity_id = entity.id

    # The `entity` relationship attribute on BaseComponent is set up to
    # load based on entity_id. Explicitly setting component_instance.entity = entity
    # links them in the ORM session immediately.
    component_instance.entity = entity

    session.add(component_instance)

    try:
        session.flush() # Flushes to DB to get component ID and catch constraint violations
    except Exception as e:
        # session.rollback() # Rollback might be too aggressive here, caller should manage transaction
        raise e # Re-raise after potential logging or specific error handling

    return component_instance


# Example of a generic component getter (can be expanded later)
# def get_components(session: Session, entity_id: int, component_type: Type[T]) -> List[T]:
#     """Retrieves all components of a specific type for a given entity."""
#     stmt = select(component_type).where(component_type.entity_id == entity_id)
#     return session.execute(stmt).scalars().all()
