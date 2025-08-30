import logging  # Added import
from typing import Any, Dict, List, Optional, Type, TypeVar  # Added Dict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession  # Import AsyncSession for type hints

# Corrected imports for BaseComponent and Entity
from dam.models.core.base_component import REGISTERED_COMPONENT_TYPES, BaseComponent
from dam.models.core.entity import Entity

# No longer need to import specific components if REGISTERED_COMPONENT_TYPES is comprehensive

# Define a generic type variable for component types
T = TypeVar("T", bound=BaseComponent)


# Note: All functions in this service require a `session: Session` argument.
# The caller (e.g., a system, another service, or a CLI command handler)
# is responsible for obtaining the correct session for the desired ECS World,
# typically by calling `world.get_db_session()`.

logger = logging.getLogger(__name__)  # Added logger


async def create_entity(session: AsyncSession) -> Entity:  # Made async, use AsyncSession
    """
    Creates a new Entity instance in the given session, adds it, and flushes.
    The caller is responsible for committing the session.
    """
    entity = Entity()
    session.add(entity)
    await session.flush()  # Await flush
    return entity


async def get_entity(session: AsyncSession, entity_id: int) -> Optional[Entity]:  # Made async, use AsyncSession
    """
    Retrieves an entity by its ID from the given session.
    Note: Generic eager loading of all 'components' via Entity.components was removed
    as the direct relationship to abstract BaseComponent was problematic.
    Components should be loaded as needed using get_component(s).
    """
    stmt = select(Entity).where(Entity.id == entity_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def add_component_to_entity(
    session: AsyncSession, entity_id: int, component_instance: T, flush: bool = True
) -> T:  # Made async
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
    entity = await get_entity(session, entity_id)  # Await async call
    if not entity:
        raise ValueError(f"Entity with ID {entity_id} not found in the provided session.")

    component_instance.entity_id = entity.id
    component_instance.entity = entity  # Link in ORM

    session.add(component_instance)

    if flush:
        try:
            await session.flush()  # Await flush
        except Exception as e:
            # Caller should manage transaction (commit/rollback)
            raise e
    return component_instance


async def get_component(session: AsyncSession, entity_id: int, component_type: Type[T]) -> Optional[T]:  # Made async
    """
    Retrieves a single component of a specific type for an entity from the given session.
    """
    stmt = select(component_type).where(component_type.entity_id == entity_id)
    result = await session.execute(stmt)  # Await execute
    return result.scalar_one_or_none()


async def get_components(session: AsyncSession, entity_id: int, component_type: Type[T]) -> List[T]:  # Made async
    """
    Retrieves all components of a specific type for an entity from the given session.
    """
    stmt = select(component_type).where(component_type.entity_id == entity_id)
    result = await session.execute(stmt)  # Await execute
    return result.scalars().all()


async def get_all_components_for_entity(session: AsyncSession, entity_id: int) -> List[BaseComponent]:
    """
    Retrieves all component instances associated with a given entity_id,
    checking against all REGISTERED_COMPONENT_TYPES.
    """
    all_components: List[BaseComponent] = []
    if not await get_entity(session, entity_id):  # Check if entity exists
        logger.warning(f"Entity with ID {entity_id} not found when trying to get all its components.")
        return []  # Or raise an error, depending on desired behavior

    for component_type in REGISTERED_COMPONENT_TYPES:
        components_of_type = await get_components(session, entity_id, component_type)
        all_components.extend(components_of_type)
    return all_components


async def get_all_components_for_entity_as_dict(
    session: AsyncSession, entity_id: int
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves all components for a given entity and returns them as a dictionary.
    """
    components = await get_all_components_for_entity(session, entity_id)
    component_dict: Dict[str, List[Dict[str, Any]]] = {}
    for component in components:
        component_name = component.__class__.__name__
        if component_name not in component_dict:
            component_dict[component_name] = []

        component_data = {}
        for c in component.__mapper__.column_attrs:
            value = getattr(component, c.key)
            if isinstance(value, bytes):
                value = value.hex()
            elif hasattr(value, "isoformat"):  # Handle datetime objects
                value = value.isoformat()
            component_data[c.key] = value

        component_dict[component_name].append(component_data)

    return component_dict


async def remove_component(session: AsyncSession, component: BaseComponent, flush: bool = False) -> None:  # Made async
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
    await session.delete(component)  # Await delete
    if flush:
        await session.flush()  # Await flush


async def delete_entity(session: AsyncSession, entity_id: int, flush: bool = True) -> bool:  # Made async
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
    entity = await get_entity(session, entity_id)  # Await async call
    if not entity:
        return False

    # Delete all associated components
    # REGISTERED_COMPONENT_TYPES should be populated correctly (e.g., in world_service or models.__init__)
    if not REGISTERED_COMPONENT_TYPES:
        # This might be a critical error or warning depending on application structure.
        # For now, we proceed, but ideally, this list is always populated.
        logger.warning(
            f"REGISTERED_COMPONENT_TYPES is empty while trying to delete entity {entity_id}. Associated components may not be fully deleted."
        )

    for component_type in REGISTERED_COMPONENT_TYPES:
        components_to_delete = await get_components(session, entity_id, component_type)  # Await async call
        for component in components_to_delete:
            # Pass flush=False as we'll do a single flush at the end if requested
            await remove_component(session, component, flush=False)  # Await async call

    await session.delete(entity)  # Await delete

    if flush:
        try:
            await session.flush()  # Await flush
        except Exception as e:
            # Caller should manage transaction
            raise e
    return True


# --- Query Helper Functions ---


async def find_entities_with_components(  # Made async
    session: AsyncSession, required_component_types: List[Type[BaseComponent]]
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

    result = await session.execute(stmt)  # Await execute
    return list(result.scalars().all())


async def find_entity_id_by_hash(
    session: AsyncSession, hash_value: str, hash_type: str = "sha256"
) -> Optional[int]:  # Use AsyncSession
    """
    Finds an entity ID by its content hash string (hex).
    Returns the Entity ID or None if not found.
    Converts hex string hash_value to bytes before querying.
    """
    from dam.models import ContentHashMD5Component, ContentHashSHA256Component  # Moved import here

    hash_bytes: bytes
    try:
        hash_bytes = bytes.fromhex(hash_value)
    except ValueError:
        logger.warning(f"Invalid hex string for hash_value: {hash_value}")
        return None

    stmt: Any  # To satisfy mypy for stmt potentially not being assigned if hash_type is invalid
    if hash_type.lower() == "sha256":
        stmt = select(ContentHashSHA256Component.entity_id).where(ContentHashSHA256Component.hash_value == hash_bytes)
    elif hash_type.lower() == "md5":
        stmt = select(ContentHashMD5Component.entity_id).where(  # type: ignore[attr-defined] # Assuming MD5 comp exists
            ContentHashMD5Component.hash_value == hash_bytes  # type: ignore[attr-defined]
        )
    else:
        logger.error(f"Unsupported hash type for find_entity_id_by_hash: {hash_type}")
        return None  # Or raise ValueError

    result = await session.execute(stmt)
    entity_id = result.scalar_one_or_none()
    return entity_id


async def get_components_by_value(  # Made async
    session: AsyncSession,
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

    result = await session.execute(stmt)  # Await execute
    return result.scalars().all()


async def find_entity_by_content_hash(
    session: AsyncSession, hash_value: bytes, hash_type: str = "sha256"
) -> Optional[Entity]:  # Made async
    """
    Finds a single entity by its content hash (SHA256 or MD5), provided as bytes.
    Returns the Entity or None if not found.
    If multiple entities somehow have the same content hash (shouldn't happen for CAS),
    it will return the first one found.
    """
    from dam.models import ContentHashMD5Component, ContentHashSHA256Component

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
    stmt = select(component_to_query).where(component_to_query.hash_value == hash_value)
    result = await session.execute(stmt)  # Await execute
    components_found = result.scalars().all()

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
        return await get_entity(session, first_component.entity_id)  # Await async call
    return None


async def find_entities_by_component_attribute_value(  # Made async
    session: AsyncSession,
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

    result = await session.execute(stmt)  # Await execute
    return list(result.scalars().all())
