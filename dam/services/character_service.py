import logging
from typing import List, Optional, Tuple

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from dam.models.conceptual import CharacterConceptComponent, EntityCharacterLinkComponent
from dam.models.core.entity import Entity
from dam.services import ecs_service

logger = logging.getLogger(__name__)


class CharacterConceptNotFoundError(Exception):
    """Custom exception for when a CharacterConcept is not found."""

    pass


class CharacterLinkNotFoundError(Exception):
    """Custom exception for when an EntityCharacterLink is not found."""

    pass


# --- Character Definition Functions ---


async def create_character_concept(
    session: AsyncSession,
    name: str,
    description: Optional[str] = None,
    # Add other character-specific fields from CharacterConceptComponent if any were added
) -> Optional[Entity]:
    """
    Creates a new character concept.
    A character concept is an entity that has a CharacterConceptComponent.
    """
    if not name:
        raise ValueError("Character name cannot be empty.")

    try:
        existing_char_concept = await get_character_concept_by_name(session, name)
        if existing_char_concept:
            logger.warning(
                f"CharacterConcept with name '{name}' already exists with Entity ID {existing_char_concept.id}."
            )
            return existing_char_concept
    except CharacterConceptNotFoundError:
        # Character does not exist, proceed to create it.
        pass

    character_entity = await ecs_service.create_entity(session)
    if character_entity.id is None:  # Should be populated after create_entity if it flushes
        await session.flush()

    # BaseConceptualInfoComponent fields (concept_name, concept_description)
    # are used for the character's name and description.
    character_concept_comp = CharacterConceptComponent(
        concept_name=name,
        concept_description=description,
        # entity_id will be set by add_component_to_entity
    )
    try:
        await ecs_service.add_component_to_entity(session, character_entity.id, character_concept_comp)
        logger.info(f"Created CharacterConcept Entity ID {character_entity.id} with name '{name}'.")
        return character_entity
    except IntegrityError:  # Catch potential unique constraint violations if name is made unique in DB
        await session.rollback()
        logger.error(
            f"Failed to create CharacterConcept '{name}' due to unique constraint violation (name likely exists)."
        )
        # Re-fetch just in case it was a race condition, though less likely with the initial check
        try:
            return await get_character_concept_by_name(session, name)
        except CharacterConceptNotFoundError:
            return None  # Truly failed
    except Exception as e:
        await session.rollback()
        logger.error(f"Unexpected error creating CharacterConcept '{name}': {e}", exc_info=True)
        return None


async def get_character_concept_by_name(session: AsyncSession, name: str) -> Entity:
    """Retrieves a character concept entity by its name."""
    stmt = (
        select(Entity)
        .join(CharacterConceptComponent, Entity.id == CharacterConceptComponent.entity_id)
        .where(CharacterConceptComponent.concept_name == name)  # Using concept_name from base class
    )
    result = await session.execute(stmt)
    entity = result.scalar_one_or_none()
    if entity is None:
        raise CharacterConceptNotFoundError(f"Character concept with name '{name}' not found.")
    return entity


async def get_character_concept_by_id(session: AsyncSession, character_concept_entity_id: int) -> Optional[Entity]:
    """Retrieves a character concept entity by its ID."""
    character_concept_entity = await ecs_service.get_entity(session, character_concept_entity_id)
    if character_concept_entity and await ecs_service.get_component(
        session, character_concept_entity_id, CharacterConceptComponent
    ):
        return character_concept_entity
    return None


async def find_character_concepts(session: AsyncSession, query_name: Optional[str] = None) -> List[Entity]:
    """Finds character concepts, optionally filtering by name."""
    stmt = select(Entity).join(CharacterConceptComponent, Entity.id == CharacterConceptComponent.entity_id)
    if query_name:
        stmt = stmt.where(CharacterConceptComponent.concept_name.ilike(f"%{query_name}%"))
    stmt = stmt.order_by(CharacterConceptComponent.concept_name)
    result = await session.execute(stmt)
    return result.scalars().all()


async def update_character_concept(
    session: AsyncSession,
    character_concept_entity_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    # Add other fields as needed
) -> Optional[CharacterConceptComponent]:
    """Updates an existing character concept."""
    char_concept_comp = await ecs_service.get_component(session, character_concept_entity_id, CharacterConceptComponent)
    if not char_concept_comp:
        logger.warning(f"CharacterConceptComponent not found for Entity ID {character_concept_entity_id}.")
        return None

    updated = False
    if name is not None and char_concept_comp.concept_name != name:
        # Check for name conflict before updating
        try:
            existing_char = await get_character_concept_by_name(session, name)
            if existing_char and existing_char.id != character_concept_entity_id:
                logger.error(
                    f"Cannot update character name to '{name}' as it already exists for CharacterConcept ID {existing_char.id}."
                )
                return None
        except CharacterConceptNotFoundError:
            pass  # Good, new name is not taken
        char_concept_comp.concept_name = name
        updated = True

    if description is not None and char_concept_comp.concept_description != description:
        char_concept_comp.concept_description = description
        updated = True

    # Handle other fields if added to CharacterConceptComponent

    if updated:
        try:
            session.add(char_concept_comp)  # Add to session if changes were made
            await session.flush()
            logger.info(f"Updated CharacterConceptComponent for Entity ID {character_concept_entity_id}.")
        except IntegrityError:  # Should be caught by pre-check, but as a safeguard
            await session.rollback()
            logger.error(
                f"Failed to update CharacterConcept '{char_concept_comp.concept_name}' due to unique constraint (name likely exists)."
            )
            return None
    return char_concept_comp


async def delete_character_concept(session: AsyncSession, character_concept_entity_id: int) -> bool:
    """Deletes a character concept and all its links to assets."""
    char_concept_entity = await get_character_concept_by_id(session, character_concept_entity_id)
    if not char_concept_entity:
        logger.warning(f"CharacterConcept Entity ID {character_concept_entity_id} not found for deletion.")
        return False

    # Delete all EntityCharacterLinkComponent instances that refer to this character concept
    stmt_delete_links = delete(EntityCharacterLinkComponent).where(
        EntityCharacterLinkComponent.character_concept_entity_id == character_concept_entity_id
    )
    await session.execute(stmt_delete_links)

    # Delete the character concept entity itself (which also deletes its CharacterConceptComponent)
    return await ecs_service.delete_entity(session, character_concept_entity_id)


# --- Character Application (Linking) Functions ---


async def apply_character_to_entity(
    session: AsyncSession,
    entity_id_to_link: int,
    character_concept_entity_id: int,
    role: Optional[str] = None,
) -> Optional[EntityCharacterLinkComponent]:
    """Applies (links) a character to an entity (e.g., an asset)."""
    target_entity = await ecs_service.get_entity(session, entity_id_to_link)
    if not target_entity:
        logger.error(f"Entity to link character to (ID: {entity_id_to_link}) not found.")
        return None

    character_concept_entity = await get_character_concept_by_id(session, character_concept_entity_id)
    if not character_concept_entity:
        logger.error(f"CharacterConcept Entity (ID: {character_concept_entity_id}) not found.")
        return None

    # Check if this exact link already exists
    existing_link_stmt = select(EntityCharacterLinkComponent).where(
        EntityCharacterLinkComponent.entity_id == entity_id_to_link,
        EntityCharacterLinkComponent.character_concept_entity_id == character_concept_entity_id,
        EntityCharacterLinkComponent.role_in_asset == role,  # Role is part of uniqueness
    )
    result_existing_link = await session.execute(existing_link_stmt)
    existing_link = result_existing_link.scalar_one_or_none()

    if existing_link:
        char_concept_comp = await ecs_service.get_component(
            session, character_concept_entity_id, CharacterConceptComponent
        )
        char_name = char_concept_comp.concept_name if char_concept_comp else "Unknown Character"
        logger.warning(
            f"Character '{char_name}' (Concept ID: {character_concept_entity_id}) with role '{role}' "
            f"is already linked to Entity {entity_id_to_link}. Not applying again."
        )
        return existing_link  # Return existing link

    link_comp = EntityCharacterLinkComponent(
        character_concept_entity_id=character_concept_entity_id,
        character_concept=character_concept_entity,  # Provide the related entity object
        role_in_asset=role,
    )

    try:
        await ecs_service.add_component_to_entity(session, target_entity.id, link_comp)
        # Fetch the component using ecs_service to get its name for logging
        char_concept_comp_for_log = await ecs_service.get_component(
            session, character_concept_entity_id, CharacterConceptComponent
        )
        char_name = char_concept_comp_for_log.concept_name if char_concept_comp_for_log else "Unknown Character"
        logger.info(
            f"Applied character '{char_name}' (Concept ID: {character_concept_entity_id}) "
            f"to Entity ID {entity_id_to_link} with role '{role}'."
        )
        return link_comp
    except IntegrityError:  # Should be caught by pre-check
        await session.rollback()
        char_name_fetch = await ecs_service.get_component(
            session, character_concept_entity_id, CharacterConceptComponent
        )
        char_name = char_name_fetch.concept_name if char_name_fetch else "Unknown Character"
        logger.error(
            f"Failed to apply character '{char_name}' to Entity {entity_id_to_link} (role: '{role}'). "
            "Likely duplicate application (this should have been caught by pre-check)."
        )
        return None
    except Exception as e:
        await session.rollback()
        logger.error(f"An unexpected error occurred while applying character link: {e}", exc_info=True)
        raise


async def remove_character_from_entity(
    session: AsyncSession,
    entity_id_linked: int,
    character_concept_entity_id: int,
    role: Optional[str] = None,  # Role must match to remove a specific link
) -> bool:
    """Removes a specific character link from an entity."""
    stmt = select(EntityCharacterLinkComponent).where(
        EntityCharacterLinkComponent.entity_id == entity_id_linked,
        EntityCharacterLinkComponent.character_concept_entity_id == character_concept_entity_id,
        EntityCharacterLinkComponent.role_in_asset == role,
    )
    result = await session.execute(stmt)
    link_comp_to_delete = result.scalar_one_or_none()

    if link_comp_to_delete:
        # ecs_service.remove_component expects the component instance
        await ecs_service.remove_component(session, link_comp_to_delete)  # This will delete the row
        char_name_fetch = await ecs_service.get_component(
            session, character_concept_entity_id, CharacterConceptComponent
        )
        char_name = char_name_fetch.concept_name if char_name_fetch else "Unknown Character"
        logger.info(
            f"Removed character '{char_name}' (Concept ID: {character_concept_entity_id}, Role: '{role}') "
            f"from Entity ID {entity_id_linked}."
        )
        return True

    logger.warning(
        f"Character link (Concept ID: {character_concept_entity_id}, Role: '{role}') "
        f"not found on Entity ID {entity_id_linked}."
    )
    return False


async def get_characters_for_entity(
    session: AsyncSession, entity_id_linked: int
) -> List[Tuple[Entity, Optional[str]]]:  # Returns (Character Concept Entity, role_in_asset)
    """Gets all characters linked to a specific entity, along with their roles."""
    stmt = select(
        EntityCharacterLinkComponent.character_concept_entity_id, EntityCharacterLinkComponent.role_in_asset
    ).where(EntityCharacterLinkComponent.entity_id == entity_id_linked)
    result = await session.execute(stmt)

    character_links_info = []
    for char_concept_id, role in result.all():
        char_concept_entity = await get_character_concept_by_id(session, char_concept_id)
        if char_concept_entity:
            character_links_info.append((char_concept_entity, role))
    return character_links_info


async def get_entities_for_character(
    session: AsyncSession,
    character_concept_entity_id: int,
    role_filter: Optional[str] = None,
    filter_by_role_presence: Optional[bool] = None,  # True for any role, False for no role (NULL)
) -> List[Entity]:
    """Gets all entities linked to a specific character concept, optionally filtering by role."""
    # First, ensure the character concept entity is valid
    char_concept_entity = await get_character_concept_by_id(session, character_concept_entity_id)
    if not char_concept_entity:
        raise CharacterConceptNotFoundError(f"CharacterConcept Entity ID {character_concept_entity_id} not found.")

    stmt = (
        select(Entity)
        .join(EntityCharacterLinkComponent, Entity.id == EntityCharacterLinkComponent.entity_id)
        .where(EntityCharacterLinkComponent.character_concept_entity_id == character_concept_entity_id)
    )
    if role_filter is not None:
        stmt = stmt.where(EntityCharacterLinkComponent.role_in_asset == role_filter)
    elif filter_by_role_presence is True:  # Entities where this character has any role
        stmt = stmt.where(EntityCharacterLinkComponent.role_in_asset.isnot(None))
    elif filter_by_role_presence is False:  # Entities where this character is linked with no specific role (NULL)
        stmt = stmt.where(EntityCharacterLinkComponent.role_in_asset.is_(None))

    stmt = stmt.distinct()  # Ensure each entity is listed once if multiple links match (e.g. different roles if not filtering by specific role)
    result = await session.execute(stmt)
    return result.scalars().all()
