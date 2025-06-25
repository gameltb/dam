from typing import List, Optional, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from dam.models import (  # Moved from below for delete_entity
    BaseComponent,
    ContentHashMD5Component,  # Replaced ContentHashComponent
    ContentHashSHA256Component,  # Replaced ContentHashComponent
    Entity,
    FileLocationComponent,
    FilePropertiesComponent,
    ImagePerceptualAHashComponent,  # Replaced ImagePerceptualHashComponent
    ImagePerceptualDHashComponent,  # Replaced ImagePerceptualHashComponent
    ImagePerceptualPHashComponent,  # Replaced ImagePerceptualHashComponent
    # Add other component types here as they are created
)

# Define a generic type variable for component types
T = TypeVar("T", bound=BaseComponent)

# Placeholder for future ECS service functions.


def create_entity(session: Session) -> Entity:
    """
    Creates a new Entity instance, adds it to the session, and flushes.
    The caller is responsible for committing the session.

    Args:
        session: The SQLAlchemy session.

    Returns:
        The newly created Entity instance (with an ID assigned after flush).
    """
    entity = Entity()  # kw_only=True means no args needed if all fields are init=False
    session.add(entity)
    session.flush()  # Assigns ID to entity
    return entity


def get_entity(session: Session, entity_id: int) -> Optional[Entity]:
    """Retrieves an entity by its ID."""
    return session.get(Entity, entity_id)


def add_component_to_entity(session: Session, entity_id: int, component_instance: T) -> T:
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
        session.flush()  # Flushes to DB to get component ID and catch constraint violations
    except Exception as e:
        # session.rollback() # Rollback might be too aggressive here, caller should manage transaction
        raise e  # Re-raise after potential logging or specific error handling

    return component_instance


def get_component(session: Session, entity_id: int, component_type: Type[T]) -> Optional[T]:
    """
    Retrieves a single component of a specific type for a given entity.
    Assumes the component is unique for the entity (e.g., FilePropertiesComponent).

    Args:
        session: The SQLAlchemy session.
        entity_id: The ID of the entity.
        component_type: The class of the component to retrieve (e.g., FilePropertiesComponent).

    Returns:
        The component instance if found, otherwise None.
    """
    stmt = select(component_type).where(component_type.entity_id == entity_id)
    return session.execute(stmt).scalar_one_or_none()


def get_components(session: Session, entity_id: int, component_type: Type[T]) -> List[T]:
    """
    Retrieves all components of a specific type for a given entity.

    Args:
        session: The SQLAlchemy session.
        entity_id: The ID of the entity.
        component_type: The class of the component to retrieve (e.g., FileLocationComponent).

    Returns:
        A list of component instances, which may be empty.
    """
    stmt = select(component_type).where(component_type.entity_id == entity_id)
    return session.execute(stmt).scalars().all()


def remove_component(session: Session, component: BaseComponent) -> None:
    """
    Deletes a specific component instance from the database.
    The caller is responsible for committing the session.

    Args:
        session: The SQLAlchemy session.
        component: The component instance to delete.
                   It must be an instance already associated with the session
                   (e.g., retrieved from the DB or added and flushed).

    Raises:
        ValueError: If the component is not a valid instance or not found in session for deletion.
    """
    if not isinstance(component, BaseComponent) or component.id is None:
        # Or handle more gracefully depending on how strict we want to be
        raise ValueError(
            "Invalid component instance provided for removal. Must be a session-managed BaseComponent with an ID."
        )

    # Ensure the component is part of the session if it's detached
    # Though usually, one would pass an object that was just fetched.
    if component not in session:
        # This can happen if component was created, committed, and then session closed & reopened.
        # Re-attaching might be needed, or fetching it first. For simplicity, assume it's managed.
        # A more robust version might try session.get(type(component), component.id) first.
        pass  # Assuming component is already part of the current session or will be handled by delete

    session.delete(component)
    # No flush here, let caller manage transaction boundaries unless specified.
    # session.flush()


# Import all known component types for delete_entity

# List of all component types to iterate over when deleting an entity
ALL_COMPONENT_TYPES = [
    ContentHashMD5Component,
    ContentHashSHA256Component,
    ImagePerceptualAHashComponent,
    ImagePerceptualDHashComponent,
    ImagePerceptualPHashComponent,
    FileLocationComponent,
    FilePropertiesComponent,
    # Add other component types here
]


def delete_entity(session: Session, entity_id: int) -> bool:
    """
    Deletes an entity and all its associated components.
    The caller is responsible for committing the session.

    Args:
        session: The SQLAlchemy session.
        entity_id: The ID of the entity to delete.

    Returns:
        True if the entity was found and deleted, False otherwise.
    """
    entity = get_entity(session, entity_id)
    if not entity:
        return False

    # Delete all associated components
    for component_type in ALL_COMPONENT_TYPES:
        components_to_delete = get_components(session, entity_id, component_type)
        for component in components_to_delete:
            remove_component(session, component)  # Uses session.delete(component)

    # Delete the entity itself
    session.delete(entity)
    # No flush here, let caller manage transaction.
    # session.flush()

    return True
