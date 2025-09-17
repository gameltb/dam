import inspect
import logging
from typing import List, Optional, Tuple, Type

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
from dam.models.core.base_component import REGISTERED_COMPONENT_TYPES, BaseComponent
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
    scope_detail: Optional[str] = None,
    description: Optional[str] = None,
    allow_values: bool = False,
) -> Optional[Entity]:
    if not tag_name:
        raise ValueError("Tag name cannot be empty.")
    if not scope_type:
        raise ValueError("Tag scope type cannot be empty.")

    try:
        existing_tag_concept = await get_tag_concept_by_name(session, tag_name)  # Await
        # If get_tag_concept_by_name returns (doesn't raise), the tag exists.
        logger.warning(f"TagConcept with name '{tag_name}' already exists with Entity ID {existing_tag_concept.id}.")
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
        logger.info(f"Created TagConcept Entity ID {tag_concept_entity.id} with name '{tag_name}'.")
        return tag_concept_entity
    except IntegrityError:
        await session.rollback()  # Await
        logger.error(
            f"Failed to create TagConcept '{tag_name}' due to unique constraint violation (name likely exists)."
        )
        return None


async def get_tag_concept_by_name(session: AsyncSession, name: str) -> Entity:  # Made async, return type changed
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


async def get_tag_concept_by_id(session: AsyncSession, tag_concept_entity_id: int) -> Optional[Entity]:  # Made async
    tag_concept_entity = await ecs_functions.get_entity(session, tag_concept_entity_id)  # Await
    if tag_concept_entity and await ecs_functions.get_component(
        session, tag_concept_entity_id, TagConceptComponent
    ):  # Await
        return tag_concept_entity
    return None


async def find_tag_concepts(  # Made async
    session: AsyncSession, query_name: Optional[str] = None, scope_type: Optional[str] = None
) -> List[Entity]:
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
    name: Optional[str] = None,
    scope_type: Optional[str] = None,
    scope_detail: Optional[str] = None,
    description: Optional[str] = None,
    allow_values: Optional[bool] = None,
) -> Optional[TagConceptComponent]:
    tag_concept_comp = await ecs_functions.get_component(session, tag_concept_entity_id, TagConceptComponent)  # Await
    if not tag_concept_comp:
        logger.warning(f"TagConceptComponent not found for Entity ID {tag_concept_entity_id}.")
        return None

    updated = False
    if name is not None and tag_concept_comp.tag_name != name:
        try:
            existing_tag = await get_tag_concept_by_name(session, name)  # Await
            if existing_tag and existing_tag.id != tag_concept_entity_id:
                logger.error(
                    f"Cannot update tag name to '{name}' as it already exists for TagConcept ID {existing_tag.id}."
                )
                return None
        except TagConceptNotFoundError:
            # This is the expected case if the new name is available
            pass
        tag_concept_comp.tag_name = name
        updated = True
    if scope_type is not None and tag_concept_comp.tag_scope_type != scope_type.upper():
        tag_concept_comp.tag_scope_type = scope_type.upper()
        updated = True

    if scope_detail == "__CLEAR__":
        if tag_concept_comp.tag_scope_detail is not None:
            tag_concept_comp.tag_scope_detail = None
            updated = True
    elif scope_detail is not None and tag_concept_comp.tag_scope_detail != scope_detail:
        tag_concept_comp.tag_scope_detail = scope_detail
        updated = True

    if description == "__CLEAR__":
        if tag_concept_comp.tag_description is not None:
            tag_concept_comp.tag_description = None
            updated = True
    elif description is not None and tag_concept_comp.tag_description != description:
        tag_concept_comp.tag_description = description
        updated = True

    if allow_values is not None and tag_concept_comp.allow_values != allow_values:
        tag_concept_comp.allow_values = allow_values
        updated = True

    if updated:
        try:
            session.add(tag_concept_comp)
            await session.flush()  # Await
            logger.info(f"Updated TagConceptComponent for Entity ID {tag_concept_entity_id}.")
        except IntegrityError:
            await session.rollback()  # Await
            logger.error(
                f"Failed to update TagConcept '{tag_concept_comp.tag_name}' due to unique constraint violation (name likely exists)."
            )
            return None
    return tag_concept_comp


async def delete_tag_concept(session: AsyncSession, tag_concept_entity_id: int) -> bool:  # Made async
    tag_concept_entity = await get_tag_concept_by_id(session, tag_concept_entity_id)  # Await
    if not tag_concept_entity:
        logger.warning(f"TagConcept Entity ID {tag_concept_entity_id} not found for deletion.")
        return False

    stmt = delete(EntityTagLinkComponent).where(EntityTagLinkComponent.tag_concept_entity_id == tag_concept_entity_id)
    await session.execute(stmt)  # Await

    return await ecs_functions.delete_entity(session, tag_concept_entity_id)  # Await


# --- Tag Application Functions ---


async def _is_scope_valid(
    session: AsyncSession, entity_id_to_tag: int, tag_concept_comp: TagConceptComponent
) -> bool:  # Made async
    # print(f"DEBUG: _is_scope_valid called for entity {entity_id_to_tag}, tag '{tag_concept_comp.tag_name}'")
    # print(f"DEBUG: REGISTERED_COMPONENT_TYPES at start of _is_scope_valid: {[c.__name__ for c in REGISTERED_COMPONENT_TYPES]}")

    scope_type = tag_concept_comp.tag_scope_type
    scope_detail = tag_concept_comp.tag_scope_detail

    if scope_type == "GLOBAL":
        return True

    if scope_type == "COMPONENT_CLASS_REQUIRED":
        if not scope_detail:
            logger.error(
                f"TagConcept '{tag_concept_comp.tag_name}' (ID: {tag_concept_comp.entity_id}) has scope COMPONENT_CLASS_REQUIRED but no scope_detail (component class name)."
            )
            return False

        required_class: Optional[Type[BaseComponent]] = None
        for comp_class in REGISTERED_COMPONENT_TYPES:
            if comp_class.__name__ == scope_detail:
                required_class = comp_class
                break

        if not required_class:
            logger.error(
                f"Scope validation failed: Component class '{scope_detail}' for tag '{tag_concept_comp.tag_name}' is not a registered component type."
            )
            return False

        if not await ecs_functions.get_component(session, entity_id_to_tag, required_class):  # Await
            logger.warning(
                f"Scope validation failed: Entity {entity_id_to_tag} does not have required component '{scope_detail}' for tag '{tag_concept_comp.tag_name}'."
            )
            return False
        return True

    if scope_type == "CONCEPTUAL_ASSET_LOCAL":
        if not scope_detail:
            logger.error(
                f"TagConcept '{tag_concept_comp.tag_name}' (ID: {tag_concept_comp.entity_id}) has scope CONCEPTUAL_ASSET_LOCAL but no scope_detail (conceptual asset entity ID)."
            )
            return False
        try:
            conceptual_asset_entity_id_for_scope = int(scope_detail)
        except ValueError:
            logger.error(
                f"Invalid scope_detail '{scope_detail}' for CONCEPTUAL_ASSET_LOCAL scope of tag '{tag_concept_comp.tag_name}'. Expected integer Entity ID."
            )
            return False

        scope_owner_is_conceptual_asset = False
        if await ecs_functions.get_component(
            session, conceptual_asset_entity_id_for_scope, ComicBookConceptComponent
        ):  # Await
            scope_owner_is_conceptual_asset = True
        else:
            for comp_type_check in REGISTERED_COMPONENT_TYPES:
                if (
                    inspect.isclass(comp_type_check)
                    and issubclass(comp_type_check, BaseConceptualInfoComponent)
                    and not comp_type_check.__dict__.get("__abstract__", False)
                ):  # Use __dict__.get for direct check
                    if await ecs_functions.get_component(
                        session, conceptual_asset_entity_id_for_scope, comp_type_check
                    ):  # Await
                        scope_owner_is_conceptual_asset = True
                        break

        if not scope_owner_is_conceptual_asset:
            logger.error(
                f"Scope detail ID {conceptual_asset_entity_id_for_scope} for tag '{tag_concept_comp.tag_name}' (scope type CONCEPTUAL_ASSET_LOCAL) does not refer to a valid conceptual asset entity."
            )
            return False

        if entity_id_to_tag == conceptual_asset_entity_id_for_scope:
            return True
        else:
            is_valid_variant = False
            for comp_type_check in REGISTERED_COMPONENT_TYPES:
                is_class = inspect.isclass(comp_type_check)
                is_variant_subclass = (
                    issubclass(comp_type_check, (BaseVariantInfoComponent, UniqueBaseVariantInfoComponent))
                    if is_class
                    else False
                )
                # Correctly check if the class itself is concrete (not inheriting __abstract__ from parent)
                is_actually_concrete = not comp_type_check.__dict__.get("__abstract__", False) if is_class else False

                if is_class and is_variant_subclass and is_actually_concrete:
                    variant_comp = await ecs_functions.get_component(session, entity_id_to_tag, comp_type_check)
                    if variant_comp:
                        actual_conceptual_id = getattr(variant_comp, "conceptual_entity_id", None)
                        if actual_conceptual_id == conceptual_asset_entity_id_for_scope:
                            is_valid_variant = True
                            break

            if is_valid_variant:
                return True

        logger.warning(
            f"Scope validation failed: Entity {entity_id_to_tag} is not the specified conceptual asset {conceptual_asset_entity_id_for_scope} nor its variant for tag '{tag_concept_comp.tag_name}'."
        )
        return False

    logger.warning(
        f"Unknown tag_scope_type '{scope_type}' for tag '{tag_concept_comp.tag_name}'. Denying application by default for unknown scopes."
    )
    return False


async def apply_tag_to_entity(  # Made async
    session: AsyncSession, entity_id_to_tag: int, tag_concept_entity_id: int, value: Optional[str] = None
) -> Optional[EntityTagLinkComponent]:
    target_entity = await ecs_functions.get_entity(session, entity_id_to_tag)  # Await
    if not target_entity:
        logger.error(f"Entity to tag (ID: {entity_id_to_tag}) not found.")
        return None

    tag_concept_entity = await get_tag_concept_by_id(session, tag_concept_entity_id)  # Await
    if not tag_concept_entity:
        logger.error(f"TagConcept Entity (ID: {tag_concept_entity_id}) not found.")
        return None

    tag_concept_comp = await ecs_functions.get_component(session, tag_concept_entity_id, TagConceptComponent)  # Await
    if not tag_concept_comp:
        logger.error(f"TagConceptComponent missing on Entity ID {tag_concept_entity_id}.")
        return None

    if not await _is_scope_valid(session, entity_id_to_tag, tag_concept_comp):  # Await
        return None

    if not tag_concept_comp.allow_values and value is not None:
        logger.warning(
            f"TagConcept '{tag_concept_comp.tag_name}' (ID: {tag_concept_entity_id}) does not allow values, but value '{value}' provided. Value will be ignored."
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
            f"Tag '{tag_concept_comp.tag_name}' with value '{value}' already applied to Entity {entity_id_to_tag}. Not applying again."
        )
        return None

    # Instantiate EntityTagLinkComponent without 'entity' (from BaseComponent, init=False)
    # 'tag_concept' is a relationship on EntityTagLinkComponent itself and is an init argument.
    link_comp = EntityTagLinkComponent(tag_concept=tag_concept_entity, tag_value=value)

    try:
        # Use ecs_functions to add the component and handle associations for BaseComponent fields
        await ecs_functions.add_component_to_entity(session, target_entity.id, link_comp)
        logger.info(
            f"Applied tag '{tag_concept_comp.tag_name}' (Concept ID: {tag_concept_entity_id}) to Entity ID {entity_id_to_tag} with value '{value}'."
        )
        return link_comp
    except IntegrityError:
        await session.rollback()  # Await
        logger.error(
            f"Failed to apply tag '{tag_concept_comp.tag_name}' to Entity {entity_id_to_tag} (value: '{value}'). Likely duplicate application (this should have been caught by pre-check)."
        )
        return None
    except Exception as e:
        await session.rollback()  # Await
        logger.error(f"An unexpected error occurred while applying tag: {e}")
        raise


async def remove_tag_from_entity(  # Made async
    session: AsyncSession, entity_id_tagged: int, tag_concept_entity_id: int, value: Optional[str] = None
) -> bool:
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
            f"Removed tag (Concept ID: {tag_concept_entity_id}, Value: '{value}') from Entity ID {entity_id_tagged}."
        )
        return True
    logger.warning(
        f"Tag application (Concept ID: {tag_concept_entity_id}, Value: '{value}') not found on Entity ID {entity_id_tagged}."
    )
    return False


async def get_tags_for_entity(
    session: AsyncSession, entity_id_tagged: int
) -> List[Tuple[Entity, Optional[str]]]:  # Made async
    stmt = select(EntityTagLinkComponent.tag_concept_entity_id, EntityTagLinkComponent.tag_value).where(
        EntityTagLinkComponent.entity_id == entity_id_tagged
    )
    result = await session.execute(stmt)  # Await
    results_all = result.all()

    tags_info: List[Tuple[Entity, Optional[str]]] = []
    for tag_concept_id, tag_val in results_all:
        tag_concept_e = await get_tag_concept_by_id(session, tag_concept_id)
        if tag_concept_e:
            tags_info.append((tag_concept_e, tag_val))
    return tags_info


async def get_entities_for_tag(  # Made async
    session: AsyncSession,  # Use AsyncSession
    tag_concept_entity_id: int,
    value_filter: Optional[str] = None,
    filter_by_value_presence: Optional[bool] = None,
) -> List[Entity]:
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
    scope_detail: Optional[str] = None,
    description: Optional[str] = None,  # Auto-tags might not have detailed descriptions initially
    allow_values: bool = False,  # Auto-tags are usually labels
) -> Optional[TagConceptComponent]:
    """
    Retrieves an existing TagConceptComponent by name, or creates a new one if not found.
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
            else:
                # This case should ideally not happen if get_tag_concept_by_name returned an entity
                # that is supposed to be a tag concept.
                logger.error(
                    f"TagConceptEntity {tag_entity.id} found for name '{clean_tag_name}', but it lacks TagConceptComponent."
                )
                # Fall through to attempt creation, though this indicates an issue.
                # Or, one might choose to raise an error here.
                # For robustness, trying to create if component is missing.
                pass  # Fall through to create logic below if component is missing.

    except TagConceptNotFoundError:
        logger.info(f"TagConcept '{clean_tag_name}' not found, attempting to create.")
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
        else:
            logger.error(
                f"Failed to retrieve TagConceptComponent from newly created TagConceptEntity {new_tag_entity.id} for '{clean_tag_name}'."
            )
            return None
    else:
        # Creation might have failed (e.g. race condition if another process created it, caught by create_tag_concept's internal check)
        # Try one more time to get it, in case of a race condition where another call created it.
        try:
            tag_entity_after_failed_create = await get_tag_concept_by_name(session, clean_tag_name)
            if tag_entity_after_failed_create:
                return await ecs_functions.get_component(
                    session, tag_entity_after_failed_create.id, TagConceptComponent
                )
        except TagConceptNotFoundError:
            logger.error(f"Failed to create and subsequently retrieve TagConcept '{clean_tag_name}'.")
            return None
    return None  # Should be unreachable if logic is correct
