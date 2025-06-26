from typing import List, Optional, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from dam.models import BaseComponent, Entity
from dam.models.base_component import REGISTERED_COMPONENT_TYPES

# No longer need to import specific components if REGISTERED_COMPONENT_TYPES is comprehensive

# Define a generic type variable for component types
T = TypeVar("T", bound=BaseComponent)

import logging # Added import

# Note: All functions in this service require a `session: Session` argument.
# The caller (e.g., a system, another service, or a CLI command handler)
# is responsible for obtaining the correct session for the desired ECS World,
# typically by calling `world.get_db_session()`.

logger = logging.getLogger(__name__) # Added logger


def create_entity(session: Session) -> Entity:
    """
    Creates a new Entity instance in the given session, adds it, and flushes.
    The caller is responsible for committing the session.
    """
    entity = Entity()
    session.add(entity)
    session.flush()  # Assigns ID to entity
    return entity


def get_entity(session: Session, entity_id: int) -> Optional[Entity]:
    """Retrieves an entity by its ID from the given session."""
    return session.get(Entity, entity_id)


def add_component_to_entity(session: Session, entity_id: int, component_instance: T, flush: bool = True) -> T:
    """
    Adds a component instance to a specified entity in the given session.

    Args:
        session: The SQLAlchemy session for the target world.
        entity_id: The ID of the entity to add the component to.
        component_instance: An instance of a class inheriting from BaseComponent.
        flush: Whether to flush the session after adding. Defaults to True.

    Returns:
        The added component instance.
    Raises:
        ValueError: If the entity is not found.
    """
    entity = get_entity(session, entity_id)
    if not entity:
        raise ValueError(f"Entity with ID {entity_id} not found in the provided session.")

    component_instance.entity_id = entity.id
    component_instance.entity = entity  # Link in ORM

    session.add(component_instance)

    if flush:
        try:
            session.flush()  # Flushes to DB, assigns component ID, checks constraints
        except Exception as e:
            # Caller should manage transaction (commit/rollback)
            raise e
    return component_instance


def get_component(session: Session, entity_id: int, component_type: Type[T]) -> Optional[T]:
    """
    Retrieves a single component of a specific type for an entity from the given session.
    """
    stmt = select(component_type).where(component_type.entity_id == entity_id)
    return session.execute(stmt).scalar_one_or_none()


def get_components(session: Session, entity_id: int, component_type: Type[T]) -> List[T]:
    """
    Retrieves all components of a specific type for an entity from the given session.
    """
    stmt = select(component_type).where(component_type.entity_id == entity_id)
    return session.execute(stmt).scalars().all()


def remove_component(session: Session, component: BaseComponent, flush: bool = False) -> None:
    """
    Deletes a specific component instance from the database via the given session.
    The caller is responsible for committing the session.

    Args:
        session: The SQLAlchemy session for the target world.
        component: The component instance to delete.
        flush: Whether to flush the session after deletion. Defaults to False.
    Raises:
        ValueError: If the component is invalid.
    """
    if not isinstance(component, BaseComponent) or component.id is None:
        raise ValueError("Invalid component instance for removal. Must be a session-managed BaseComponent with an ID.")
    # No need to check `if component not in session` if it's fetched via the same session.
    # If it could be from another session, then `session.merge(component)` might be needed before delete.
    # For simplicity, assume it's from the current session or attached.
    session.delete(component)
    if flush:
        session.flush()


def delete_entity(session: Session, entity_id: int, flush: bool = True) -> bool:
    """
    Deletes an entity and all its associated components from the given session.
    The caller is responsible for committing the session.

    Args:
        session: The SQLAlchemy session for the target world.
        entity_id: The ID of the entity to delete.
        flush: Whether to flush the session after deletions. Defaults to True.
                 Set to False if part of a larger operation to be flushed later.
    Returns:
        True if the entity was found and deleted, False otherwise.
    """
    entity = get_entity(session, entity_id)
    if not entity:
        return False

    # Delete all associated components
    # REGISTERED_COMPONENT_TYPES should be populated correctly (e.g., in world_service or models.__init__)
    if not REGISTERED_COMPONENT_TYPES:
        # This might be a critical error or warning depending on application structure.
        # For now, we proceed, but ideally, this list is always populated.
        logger.warning(
            f"REGISTERED_COMPONENT_TYPES is empty while trying to delete entity {entity_id}. "
            "Associated components may not be fully deleted."
        )

    for component_type in REGISTERED_COMPONENT_TYPES:
        components_to_delete = get_components(session, entity_id, component_type)
        for component in components_to_delete:
            # Pass flush=False as we'll do a single flush at the end if requested
            remove_component(session, component, flush=False)

    session.delete(entity)

    if flush:
        try:
            session.flush()
        except Exception as e:
            # Caller should manage transaction
            raise e
    return True
