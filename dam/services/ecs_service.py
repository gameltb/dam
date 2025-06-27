import logging  # Added import
from typing import Any, Dict, List, Optional, Type, TypeVar  # Added Dict

from sqlalchemy import select
from sqlalchemy.orm import Session

from dam.models import BaseComponent, Entity
from dam.models.base_component import REGISTERED_COMPONENT_TYPES

# No longer need to import specific components if REGISTERED_COMPONENT_TYPES is comprehensive

# Define a generic type variable for component types
T = TypeVar("T", bound=BaseComponent)


# Note: All functions in this service require a `session: Session` argument.
# The caller (e.g., a system, another service, or a CLI command handler)
# is responsible for obtaining the correct session for the desired ECS World,
# typically by calling `world.get_db_session()`.

logger = logging.getLogger(__name__)  # Added logger


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


# --- Query Helper Functions ---


def find_entities_with_components(
    session: Session, required_component_types: List[Type[BaseComponent]]
) -> List[Entity]:
    """
    Finds entities that have ALL of the specified component types.
    Entities are returned distinct.
    """
    if not required_component_types:
        return []

    stmt = select(Entity)
    for i, comp_type in enumerate(required_component_types):
        if not issubclass(comp_type, BaseComponent):
            raise TypeError(f"Type {comp_type} is not a BaseComponent subclass.")
        # Use aliasing if the same component type could be required multiple times with different criteria
        # (not applicable here, but good practice if extending for attribute checks on each).
        # For now, a simple join is fine. If a component could appear multiple times for an entity
        # and we only care about its presence, distinct on entity_id from component or a subquery is safer.
        # However, typical ECS components are one-per-entity or handled by systems aware of multiples.
        # This join assumes we just need to ensure each type is present.
        stmt = stmt.join(comp_type, Entity.id == comp_type.entity_id)

    # Ensure distinct entities if multiple components of the same type or complex joins might cause duplicates
    stmt = stmt.distinct()

    result = session.execute(stmt).scalars().all()
    return list(result)


def get_components_by_value(
    session: Session,
    entity_id: int,
    component_type: Type[T],
    attributes_values: Dict[str, Any],
) -> List[T]:
    """
    Retrieves components of a specific type for an entity that match all given attribute values.
    """
    if not issubclass(component_type, BaseComponent):
        raise TypeError(f"Type {component_type} is not a BaseComponent subclass.")

    stmt = select(component_type).where(component_type.entity_id == entity_id)
    for attr_name, value in attributes_values.items():
        if not hasattr(component_type, attr_name):
            raise AttributeError(f"Component {component_type.__name__} has no attribute '{attr_name}'.")
        stmt = stmt.where(getattr(component_type, attr_name) == value)

    return session.execute(stmt).scalars().all()


def find_entity_by_content_hash(session: Session, hash_value: bytes, hash_type: str = "sha256") -> Optional[Entity]:
    """
    Finds a single entity by its content hash (SHA256 or MD5), provided as bytes.
    Returns the Entity or None if not found.
    If multiple entities somehow have the same content hash (shouldn't happen for CAS),
    it will return the first one found.
    """
    from dam.models.content_hash_md5_component import ContentHashMD5Component
    from dam.models.content_hash_sha256_component import ContentHashSHA256Component

    component_to_query: Type[BaseComponent]
    if hash_type.lower() == "sha256":
        component_to_query = ContentHashSHA256Component
    elif hash_type.lower() == "md5":
        component_to_query = ContentHashMD5Component
    else:
        logger.warning(f"Unsupported hash_type '{hash_type}' for find_entity_by_content_hash.")
        return None

    # Use get_components_by_value to find matching components first.
    # An entity ID isn't known yet, so we can't use entity_id filter in get_components_by_value.
    # We need a broader query for components matching the hash_value across all entities.

    # Simpler: query component_to_query directly.
    stmt = select(component_to_query).where(getattr(component_to_query, "hash_value") == hash_value)
    components_found = session.execute(stmt).scalars().all()

    if components_found:
        if len(components_found) > 1:
            # This case (multiple distinct entities having components with the exact same hash_value)
            # should ideally not happen if hashes are unique identifiers for content.
            # Log a warning if it does.
            entity_ids = sorted(list(set(c.entity_id for c in components_found)))
            logger.warning(
                f"Found multiple components ({len(components_found)}) matching {hash_type} hash '{hash_value}' "
                f"across different entities (IDs: {entity_ids}). This might indicate a data integrity issue "
                f"if content hashes are expected to be unique per entity. Returning entity of the first component found."
            )
        # Return the parent Entity of the first found component.
        first_component = components_found[0]
        return get_entity(session, first_component.entity_id)  # Fetch the entity
    return None


def find_entities_by_component_attribute_value(
    session: Session,
    component_type: Type[T],
    attribute_name: str,
    value: Any,
    # TODO: Consider adding options for specific SQLAlchemy relationship loading for Entity (e.g. using options())
    # to allow preloading other components of the found entities.
) -> List[Entity]:
    """
    Finds entities that have a component of `component_type`
    where `component_type.attribute_name == value`.
    """
    if not issubclass(component_type, BaseComponent):
        raise TypeError(f"Type {component_type} is not a BaseComponent subclass.")
    if not hasattr(component_type, attribute_name):
        raise AttributeError(f"Component {component_type.__name__} has no attribute '{attribute_name}'.")

    stmt = (
        select(Entity)
        .join(component_type, Entity.id == component_type.entity_id)
        .where(getattr(component_type, attribute_name) == value)
        .distinct()  # Ensure distinct entities are returned
    )

    result = session.execute(stmt).scalars().all()
    return list(result)
