"""Functions for managing tag concepts and their links to entities."""

import inspect
import logging

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession  # Import AsyncSession

from dam.functions import ecs_functions
from dam.models.conceptual import (  # Keep conceptual imports that are still relevant
    BaseConceptualInfoComponent,
    BaseVariantInfoComponent,
    ComicBookConceptComponent,
    UniqueBaseVariantInfoComponent,
)
from dam.models.core.base_component import (
    REGISTERED_COMPONENT_TYPES,
)
from dam.models.core.entity import Entity

# Updated imports for tag components
from dam.models.tags import (
    EntityTagLinkComponent,
    TagConceptComponent,
)

logger = logging.getLogger(__name__)


class TagConceptNotFoundError(Exception):
    """Custom exception for when a TagConcept is not found."""

    pass


# --- Tag Definition Functions ---


async def create_tag_concept(  # Made async
    session: AsyncSession,  # Use AsyncSession
    tag_name: str,
    scope_type: str,
    scope_detail: str | None = None,
    description: str | None = None,
    allow_values: bool = False,
) -> Entity | None:
    """Create a new tag concept."""
    if not tag_name:
        raise ValueError("Tag name cannot be empty.")
    if not scope_type:
        raise ValueError("Tag scope type cannot be empty.")

    try:
        existing_tag_concept = await get_tag_concept_by_name(session, tag_name)  # Await
        # If get_tag_concept_by_name returns (doesn't raise), the tag exists.
        logger.warning("TagConcept with name '%s' already exists with Entity ID %s.", tag_name, existing_tag_concept.id)
        return existing_tag_concept
    except TagConceptNotFoundError:
        # Tag does not exist, proceed to create it.
        pass

    tag_concept_entity = await ecs_functions.create_entity(session)  # Await

    tag_concept_comp = TagConceptComponent(
        tag_name=tag_name,
        tag_scope_type=scope_type.upper(),
        tag_scope_detail=scope_detail,
        tag_description=description,
        allow_values=allow_values,
        concept_name=tag_name,  # Use tag_name for the general concept_name
        concept_description=description,  # Use description for the general concept_description
    )
    try:
        await ecs_functions.add_component_to_entity(session, tag_concept_entity.id, tag_concept_comp)
        logger.info("Created TagConcept Entity ID %s with name '%s'.", tag_concept_entity.id, tag_name)
        return tag_concept_entity
    except IntegrityError:
        await session.rollback()  # Await
        logger.error(
            "Failed to create TagConcept '%s' due to unique constraint violation (name likely exists).", tag_name
        )
        return None


async def get_tag_concept_by_name(session: AsyncSession, name: str) -> Entity:  # Made async, return type changed
    """Retrieve a tag concept entity by its name."""
    stmt = (
        select(Entity)
        .join(TagConceptComponent, Entity.id == TagConceptComponent.entity_id)
        .where(TagConceptComponent.tag_name == name)
    )
    result = await session.execute(stmt)  # Await
    entity = result.scalar_one_or_none()
    if entity is None:
        raise TagConceptNotFoundError(f"Tag concept '{name}' not found.")
    return entity


async def get_tag_concept_by_id(session: AsyncSession, tag_concept_entity_id: int) -> Entity | None:  # Made async
    """Retrieve a tag concept entity by its ID."""
    tag_concept_entity = await ecs_functions.get_entity(session, tag_concept_entity_id)  # Await
    if tag_concept_entity and await ecs_functions.get_component(
        session, tag_concept_entity_id, TagConceptComponent
    ):  # Await
        return tag_concept_entity
    return None


async def find_tag_concepts(  # Made async
    session: AsyncSession, query_name: str | None = None, scope_type: str | None = None
) -> list[Entity]:
    """Find tag concepts, optionally filtering by name and scope type."""
    stmt = select(Entity).join(TagConceptComponent, Entity.id == TagConceptComponent.entity_id)
    if query_name:
        stmt = stmt.where(TagConceptComponent.tag_name.ilike(f"%{query_name}%"))
    if scope_type:
        stmt = stmt.where(TagConceptComponent.tag_scope_type == scope_type.upper())

    stmt = stmt.order_by(TagConceptComponent.tag_name)
    result = await session.execute(stmt)  # Await
    return list(result.scalars().all())


async def update_tag_concept(  # Made async
    session: AsyncSession,  # Use AsyncSession
    tag_concept_entity_id: int,
    name: str | None = None,
    scope_type: str | None = None,
    scope_detail: str | None = None,
    description: str | None = None,
    allow_values: bool | None = None,
) -> TagConceptComponent | None:
    """Update an existing tag concept."""
    tag_concept_comp = await ecs_functions.get_component(session, tag_concept_entity_id, TagConceptComponent)
    if not tag_concept_comp:
        logger.warning("TagConceptComponent not found for Entity ID %s.", tag_concept_entity_id)
        return None

    updated = False
    if name is not None and tag_concept_comp.tag_name != name:
        try:
            existing_tag = await get_tag_concept_by_name(session, name)
            if existing_tag and existing_tag.id != tag_concept_entity_id:
                logger.error(
                    "Cannot update tag name to '%s' as it already exists for TagConcept ID %s.", name, existing_tag.id
                )
                return None
        except TagConceptNotFoundError:
            pass  # This is the expected case if the new name is available
        tag_concept_comp.tag_name = name
        updated = True

    if scope_type is not None and tag_concept_comp.tag_scope_type != scope_type.upper():
        tag_concept_comp.tag_scope_type = scope_type.upper()
        updated = True

    if scope_detail is not None:
        new_scope_detail = None if scope_detail == "__CLEAR__" else scope_detail
        if tag_concept_comp.tag_scope_detail != new_scope_detail:
            tag_concept_comp.tag_scope_detail = new_scope_detail
            updated = True

    if description is not None:
        new_description = None if description == "__CLEAR__" else description
        if tag_concept_comp.tag_description != new_description:
            tag_concept_comp.tag_description = new_description
            updated = True

    if allow_values is not None and tag_concept_comp.allow_values != allow_values:
        tag_concept_comp.allow_values = allow_values
        updated = True

    if updated:
        try:
            session.add(tag_concept_comp)
            await session.flush()
            logger.info("Updated TagConceptComponent for Entity ID %s.", tag_concept_entity_id)
        except IntegrityError:
            await session.rollback()
            logger.error(
                "Failed to update TagConcept '%s' due to unique constraint violation (name likely exists).",
                tag_concept_comp.tag_name,
            )
            return None
    return tag_concept_comp


async def delete_tag_concept(session: AsyncSession, tag_concept_entity_id: int) -> bool:  # Made async
    """Delete a tag concept and all its links to entities."""
    tag_concept_entity = await get_tag_concept_by_id(session, tag_concept_entity_id)
    if not tag_concept_entity:
        logger.warning("TagConcept Entity ID %s not found for deletion.", tag_concept_entity_id)
        return False

    stmt = delete(EntityTagLinkComponent).where(EntityTagLinkComponent.tag_concept_entity_id == tag_concept_entity_id)
    await session.execute(stmt)

    return await ecs_functions.delete_entity(session, tag_concept_entity_id)


# --- Tag Application Functions ---


async def _validate_scope_component_class(
    session: AsyncSession, entity_id_to_tag: int, tag_concept_comp: TagConceptComponent
) -> bool:
    """Validate scope for COMPONENT_CLASS_REQUIRED."""
    scope_detail = tag_concept_comp.tag_scope_detail
    if not scope_detail:
        logger.error(
            "TagConcept '%s' (ID: %s) has scope COMPONENT_CLASS_REQUIRED but no scope_detail.",
            tag_concept_comp.tag_name,
            tag_concept_comp.entity_id,
        )
        return False

    required_class = next((c for c in REGISTERED_COMPONENT_TYPES if c.__name__ == scope_detail), None)
    if not required_class:
        logger.error(
            "Scope validation failed: Component class '%s' for tag '%s' is not registered.",
            scope_detail,
            tag_concept_comp.tag_name,
        )
        return False

    if not await ecs_functions.get_component(session, entity_id_to_tag, required_class):
        logger.warning(
            "Scope validation failed: Entity %s lacks required component '%s' for tag '%s'.",
            entity_id_to_tag,
            scope_detail,
            tag_concept_comp.tag_name,
        )
        return False
    return True


async def _is_conceptual_asset(session: AsyncSession, entity_id: int) -> bool:
    """Check if an entity is a conceptual asset."""
    if await ecs_functions.get_component(session, entity_id, ComicBookConceptComponent):
        return True
    for comp_type in REGISTERED_COMPONENT_TYPES:
        if (
            inspect.isclass(comp_type)
            and issubclass(comp_type, BaseConceptualInfoComponent)
            and not comp_type.__dict__.get("__abstract__", False)
            and await ecs_functions.get_component(session, entity_id, comp_type)
        ):
            return True
    return False


async def _is_variant_of_conceptual_asset(
    session: AsyncSession, variant_entity_id: int, conceptual_entity_id: int
) -> bool:
    """Check if an entity is a variant of a given conceptual asset."""
    for comp_type in REGISTERED_COMPONENT_TYPES:
        is_class = inspect.isclass(comp_type)
        is_variant = is_class and issubclass(comp_type, (BaseVariantInfoComponent, UniqueBaseVariantInfoComponent))
        is_concrete = is_class and not comp_type.__dict__.get("__abstract__", False)

        if is_variant and is_concrete:
            variant_comp = await ecs_functions.get_component(session, variant_entity_id, comp_type)
            if variant_comp and getattr(variant_comp, "conceptual_entity_id", None) == conceptual_entity_id:
                return True
    return False


async def _validate_scope_conceptual_asset_local(
    session: AsyncSession, entity_id_to_tag: int, tag_concept_comp: TagConceptComponent
) -> bool:
    """Validate scope for CONCEPTUAL_ASSET_LOCAL."""
    scope_detail = tag_concept_comp.tag_scope_detail
    if not scope_detail:
        logger.error(
            "TagConcept '%s' (ID: %s) has scope CONCEPTUAL_ASSET_LOCAL but no scope_detail.",
            tag_concept_comp.tag_name,
            tag_concept_comp.entity_id,
        )
        return False
    try:
        conceptual_id = int(scope_detail)
    except ValueError:
        logger.error(
            "Invalid scope_detail '%s' for tag '%s'. Expected integer Entity ID.",
            scope_detail,
            tag_concept_comp.tag_name,
        )
        return False

    if not await _is_conceptual_asset(session, conceptual_id):
        logger.error(
            "Scope detail ID %s for tag '%s' is not a valid conceptual asset.", conceptual_id, tag_concept_comp.tag_name
        )
        return False

    if entity_id_to_tag == conceptual_id or await _is_variant_of_conceptual_asset(
        session, entity_id_to_tag, conceptual_id
    ):
        return True

    logger.warning(
        "Scope validation failed: Entity %s is not asset %s nor its variant for tag '%s'.",
        entity_id_to_tag,
        conceptual_id,
        tag_concept_comp.tag_name,
    )
    return False


async def _is_scope_valid(session: AsyncSession, entity_id_to_tag: int, tag_concept_comp: TagConceptComponent) -> bool:
    """Check if an entity is within the scope of a tag concept."""
    scope_type = tag_concept_comp.tag_scope_type

    if scope_type == "GLOBAL":
        return True
    if scope_type == "COMPONENT_CLASS_REQUIRED":
        return await _validate_scope_component_class(session, entity_id_to_tag, tag_concept_comp)
    if scope_type == "CONCEPTUAL_ASSET_LOCAL":
        return await _validate_scope_conceptual_asset_local(session, entity_id_to_tag, tag_concept_comp)

    logger.warning(
        "Unknown tag_scope_type '%s' for tag '%s'. Denying application by default.",
        scope_type,
        tag_concept_comp.tag_name,
    )
    return False


async def apply_tag_to_entity(  # Made async
    session: AsyncSession, entity_id_to_tag: int, tag_concept_entity_id: int, value: str | None = None
) -> EntityTagLinkComponent | None:
    """Apply a tag to an entity, optionally with a value."""
    target_entity = await ecs_functions.get_entity(session, entity_id_to_tag)  # Await
    if not target_entity:
        logger.error("Entity to tag (ID: %s) not found.", entity_id_to_tag)
        return None

    tag_concept_entity = await get_tag_concept_by_id(session, tag_concept_entity_id)  # Await
    if not tag_concept_entity:
        logger.error("TagConcept Entity (ID: %s) not found.", tag_concept_entity_id)
        return None

    tag_concept_comp = await ecs_functions.get_component(session, tag_concept_entity_id, TagConceptComponent)  # Await
    if not tag_concept_comp:
        logger.error("TagConceptComponent missing on Entity ID %s.", tag_concept_entity_id)
        return None

    if not await _is_scope_valid(session, entity_id_to_tag, tag_concept_comp):  # Await
        return None

    if not tag_concept_comp.allow_values and value is not None:
        logger.warning(
            "TagConcept '%s' (ID: %s) does not allow values, but value '%s' provided. Value will be ignored.",
            tag_concept_comp.tag_name,
            tag_concept_entity_id,
            value,
        )
        value = None

    existing_link_stmt = select(EntityTagLinkComponent).where(
        EntityTagLinkComponent.entity_id == entity_id_to_tag,
        EntityTagLinkComponent.tag_concept_entity_id == tag_concept_entity_id,
        EntityTagLinkComponent.tag_value == value,
    )
    result_existing_link = await session.execute(existing_link_stmt)  # Await
    existing_link = result_existing_link.scalar_one_or_none()

    if existing_link:
        logger.warning(
            "Tag '%s' with value '%s' already applied to Entity %s. Not applying again.",
            tag_concept_comp.tag_name,
            value,
            entity_id_to_tag,
        )
        return None

    # Instantiate EntityTagLinkComponent without 'entity' (from BaseComponent, init=False)
    # 'tag_concept' is a relationship on EntityTagLinkComponent itself and is an init argument.
    link_comp = EntityTagLinkComponent(tag_concept=tag_concept_entity, tag_value=value)

    try:
        # Use ecs_functions to add the component and handle associations for BaseComponent fields
        await ecs_functions.add_component_to_entity(session, target_entity.id, link_comp)
        logger.info(
            "Applied tag '%s' (Concept ID: %s) to Entity ID %s with value '%s'.",
            tag_concept_comp.tag_name,
            tag_concept_entity_id,
            entity_id_to_tag,
            value,
        )
        return link_comp
    except IntegrityError:
        await session.rollback()  # Await
        logger.error(
            "Failed to apply tag '%s' to Entity %s (value: '%s'). Likely duplicate application (this should have been caught by pre-check).",
            tag_concept_comp.tag_name,
            entity_id_to_tag,
            value,
        )
        return None
    except Exception:
        await session.rollback()  # Await
        logger.exception("An unexpected error occurred while applying tag.")
        raise


async def remove_tag_from_entity(  # Made async
    session: AsyncSession, entity_id_tagged: int, tag_concept_entity_id: int, value: str | None = None
) -> bool:
    """Remove a tag from an entity."""
    stmt = select(EntityTagLinkComponent).where(
        EntityTagLinkComponent.entity_id == entity_id_tagged,
        EntityTagLinkComponent.tag_concept_entity_id == tag_concept_entity_id,
        EntityTagLinkComponent.tag_value == value,
    )
    result = await session.execute(stmt)  # Await
    link_comp_to_delete = result.scalar_one_or_none()

    if link_comp_to_delete:
        await session.delete(link_comp_to_delete)  # Await
        logger.info(
            "Removed tag (Concept ID: %s, Value: '%s') from Entity ID %s.",
            tag_concept_entity_id,
            value,
            entity_id_tagged,
        )
        return True
    logger.warning(
        "Tag application (Concept ID: %s, Value: '%s') not found on Entity ID %s.",
        tag_concept_entity_id,
        value,
        entity_id_tagged,
    )
    return False


async def get_tags_for_entity(
    session: AsyncSession, entity_id_tagged: int
) -> list[tuple[Entity, str | None]]:  # Made async
    """Get all tags for a given entity."""
    stmt = select(EntityTagLinkComponent.tag_concept_entity_id, EntityTagLinkComponent.tag_value).where(
        EntityTagLinkComponent.entity_id == entity_id_tagged
    )
    result = await session.execute(stmt)  # Await
    results_all = result.all()

    tags_info: list[tuple[Entity, str | None]] = []
    for tag_concept_id, tag_val in results_all:
        tag_concept_e = await get_tag_concept_by_id(session, tag_concept_id)
        if tag_concept_e:
            tags_info.append((tag_concept_e, tag_val))
    return tags_info


async def get_entities_for_tag(  # Made async
    session: AsyncSession,  # Use AsyncSession
    tag_concept_entity_id: int,
    value_filter: str | None = None,
    filter_by_value_presence: bool | None = None,
) -> list[Entity]:
    """Get all entities for a given tag, with optional value filtering."""
    stmt = (
        select(Entity)
        .join(EntityTagLinkComponent, Entity.id == EntityTagLinkComponent.entity_id)
        .where(EntityTagLinkComponent.tag_concept_entity_id == tag_concept_entity_id)
    )
    if value_filter is not None:
        stmt = stmt.where(EntityTagLinkComponent.tag_value == value_filter)
    elif filter_by_value_presence is True:
        stmt = stmt.where(EntityTagLinkComponent.tag_value.isnot(None))
    elif filter_by_value_presence is False:
        stmt = stmt.where(EntityTagLinkComponent.tag_value.is_(None))

    stmt = stmt.distinct()
    result = await session.execute(stmt)  # Await
    return list(result.scalars().all())


async def get_or_create_tag_concept(
    session: AsyncSession,
    tag_name: str,
    scope_type: str = "GLOBAL",  # Default for auto-generated tags, can be configured
    scope_detail: str | None = None,
    description: str | None = None,  # Auto-tags might not have detailed descriptions initially
    allow_values: bool = False,  # Auto-tags are usually labels
) -> TagConceptComponent | None:
    """
    Retrieve an existing TagConceptComponent by name, or create a new one if not found.

    Returns the TagConceptComponent instance, or None if creation fails.
    """
    clean_tag_name = tag_name.strip()
    if not clean_tag_name:
        logger.warning("Attempted to get or create a tag with an empty name.")
        return None

    try:
        tag_entity = await get_tag_concept_by_name(session, clean_tag_name)
        if tag_entity:
            # Fetch the component from the entity
            tag_concept_comp = await ecs_functions.get_component(session, tag_entity.id, TagConceptComponent)
            if tag_concept_comp:
                return tag_concept_comp
            # This case should ideally not happen if get_tag_concept_by_name returned an entity
            # that is supposed to be a tag concept.
            logger.error(
                "TagConceptEntity %s found for name '%s', but it lacks TagConceptComponent.",
                tag_entity.id,
                clean_tag_name,
            )
            # Fall through to attempt creation, though this indicates an issue.
            # Or, one might choose to raise an error here.
            # For robustness, trying to create if component is missing.
            pass  # Fall through to create logic below if component is missing.

    except TagConceptNotFoundError:
        logger.info("TagConcept '%s' not found, attempting to create.", clean_tag_name)
        pass  # Tag not found, will proceed to create

    # If not found or component was missing, create it
    # create_tag_concept already checks for existing name before creating entity and component.
    # It returns the Entity.
    new_tag_entity = await create_tag_concept(
        session,
        tag_name=clean_tag_name,
        scope_type=scope_type,
        scope_detail=scope_detail,
        description=description,
        allow_values=allow_values,
    )

    if new_tag_entity:
        # Fetch the component from the newly created entity
        new_tag_concept_comp = await ecs_functions.get_component(session, new_tag_entity.id, TagConceptComponent)
        if new_tag_concept_comp:
            return new_tag_concept_comp
        logger.error(
            "Failed to retrieve TagConceptComponent from newly created TagConceptEntity %s for '%s'.",
            new_tag_entity.id,
            clean_tag_name,
        )
        return None
    # Creation might have failed (e.g. race condition if another process created it, caught by create_tag_concept's internal check)
    # Try one more time to get it, in case of a race condition where another call created it.
    try:
        tag_entity_after_failed_create = await get_tag_concept_by_name(session, clean_tag_name)
        if tag_entity_after_failed_create:
            return await ecs_functions.get_component(session, tag_entity_after_failed_create.id, TagConceptComponent)
    except TagConceptNotFoundError:
        logger.error("Failed to create and subsequently retrieve TagConcept '%s'.", clean_tag_name)
        return None
    return None  # Should be unreachable if logic is correct
