"""Functions for managing character concepts and their links to entities."""

import logging

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from dam.functions import ecs_functions
from dam.models.conceptual import CharacterConceptComponent, EntityCharacterLinkComponent
from dam.models.core.entity import Entity

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
    description: str | None = None,
    # Add other character-specific fields from CharacterConceptComponent if any were added
) -> Entity | None:
    """
    Create a new character concept.

    A character concept is an entity that has a CharacterConceptComponent.
    """
    if not name:
        raise ValueError("Character name cannot be empty.")

    try:
        existing_char_concept = await get_character_concept_by_name(session, name)
        if existing_char_concept:
            logger.warning(
                "CharacterConcept with name '%s' already exists with Entity ID %s.",
                name,
                existing_char_concept.id,
            )
            return existing_char_concept
    except CharacterConceptNotFoundError:
        # Character does not exist, proceed to create it.
        pass

    character_entity = await ecs_functions.create_entity(session)

    # BaseConceptualInfoComponent fields (concept_name, concept_description)
    # are used for the character's name and description.
    character_concept_comp = CharacterConceptComponent(
        concept_name=name,
        concept_description=description,
        # entity_id will be set by add_component_to_entity
    )
    try:
        await ecs_functions.add_component_to_entity(session, character_entity.id, character_concept_comp)
        logger.info("Created CharacterConcept Entity ID %s with name '%s'.", character_entity.id, name)
        return character_entity
    except IntegrityError:  # Catch potential unique constraint violations if name is made unique in DB
        await session.rollback()
        logger.error(
            "Failed to create CharacterConcept '%s' due to unique constraint violation (name likely exists).", name
        )
        # Re-fetch just in case it was a race condition, though less likely with the initial check
        try:
            return await get_character_concept_by_name(session, name)
        except CharacterConceptNotFoundError:
            return None  # Truly failed
    except Exception:
        await session.rollback()
        logger.exception("Unexpected error creating CharacterConcept '%s'", name)
        return None


async def get_character_concept_by_name(session: AsyncSession, name: str) -> Entity:
    """Retrieve a character concept entity by its name."""
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


async def get_character_concept_by_id(session: AsyncSession, character_concept_entity_id: int) -> Entity | None:
    """Retrieve a character concept entity by its ID."""
    character_concept_entity = await ecs_functions.get_entity(session, character_concept_entity_id)
    if character_concept_entity and await ecs_functions.get_component(
        session, character_concept_entity_id, CharacterConceptComponent
    ):
        return character_concept_entity
    return None


async def find_character_concepts(session: AsyncSession, query_name: str | None = None) -> list[Entity]:
    """Find character concepts, optionally filtering by name."""
    stmt = select(Entity).join(CharacterConceptComponent, Entity.id == CharacterConceptComponent.entity_id)
    if query_name:
        stmt = stmt.where(CharacterConceptComponent.concept_name.ilike(f"%{query_name}%"))
    stmt = stmt.order_by(CharacterConceptComponent.concept_name)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_character_concept(
    session: AsyncSession,
    character_concept_entity_id: int,
    name: str | None = None,
    description: str | None = None,
    # Add other fields as needed
) -> CharacterConceptComponent | None:
    """Update an existing character concept."""
    char_concept_comp = await ecs_functions.get_component(
        session, character_concept_entity_id, CharacterConceptComponent
    )
    if not char_concept_comp:
        logger.warning("CharacterConceptComponent not found for Entity ID %s.", character_concept_entity_id)
        return None

    updated = False
    if name is not None and char_concept_comp.concept_name != name:
        # Check for name conflict before updating
        try:
            existing_char = await get_character_concept_by_name(session, name)
            if existing_char and existing_char.id != character_concept_entity_id:
                logger.error(
                    "Cannot update character name to '%s' as it already exists for CharacterConcept ID %s.",
                    name,
                    existing_char.id,
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
            logger.info("Updated CharacterConceptComponent for Entity ID %s.", character_concept_entity_id)
        except IntegrityError:  # Should be caught by pre-check, but as a safeguard
            await session.rollback()
            logger.error(
                "Failed to update CharacterConcept '%s' due to unique constraint (name likely exists).",
                char_concept_comp.concept_name,
            )
            return None
    return char_concept_comp


async def delete_character_concept(session: AsyncSession, character_concept_entity_id: int) -> bool:
    """Delete a character concept and all its links to assets."""
    char_concept_entity = await get_character_concept_by_id(session, character_concept_entity_id)
    if not char_concept_entity:
        logger.warning("CharacterConcept Entity ID %s not found for deletion.", character_concept_entity_id)
        return False

    # Delete all EntityCharacterLinkComponent instances that refer to this character concept
    stmt_delete_links = delete(EntityCharacterLinkComponent).where(
        EntityCharacterLinkComponent.character_concept_entity_id == character_concept_entity_id
    )
    await session.execute(stmt_delete_links)

    # Delete the character concept entity itself (which also deletes its CharacterConceptComponent)
    return await ecs_functions.delete_entity(session, character_concept_entity_id)


# --- Character Application (Linking) Functions ---


async def apply_character_to_entity(
    session: AsyncSession,
    entity_id_to_link: int,
    character_concept_entity_id: int,
    role: str | None = None,
) -> EntityCharacterLinkComponent | None:
    """Apply (link) a character to an entity (e.g., an asset)."""
    target_entity = await ecs_functions.get_entity(session, entity_id_to_link)
    if not target_entity:
        logger.error("Entity to link character to (ID: %s) not found.", entity_id_to_link)
        return None

    character_concept_entity = await get_character_concept_by_id(session, character_concept_entity_id)
    if not character_concept_entity:
        logger.error("CharacterConcept Entity (ID: %s) not found.", character_concept_entity_id)
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
        char_concept_comp = await ecs_functions.get_component(
            session, character_concept_entity_id, CharacterConceptComponent
        )
        char_name = char_concept_comp.concept_name if char_concept_comp else "Unknown Character"
        logger.warning(
            "Character '%s' (Concept ID: %s) with role '%s' is already linked to Entity %s. Not applying again.",
            char_name,
            character_concept_entity_id,
            role,
            entity_id_to_link,
        )
        return existing_link  # Return existing link

    link_comp = EntityCharacterLinkComponent(
        character_concept_entity_id=character_concept_entity_id,
        character_concept=character_concept_entity,  # Provide the related entity object
        role_in_asset=role,
    )

    try:
        await ecs_functions.add_component_to_entity(session, target_entity.id, link_comp)
        # Fetch the component using ecs_functions to get its name for logging
        char_concept_comp_for_log = await ecs_functions.get_component(
            session, character_concept_entity_id, CharacterConceptComponent
        )
        char_name = char_concept_comp_for_log.concept_name if char_concept_comp_for_log else "Unknown Character"
        logger.info(
            "Applied character '%s' (Concept ID: %s) to Entity ID %s with role '%s'.",
            char_name,
            character_concept_entity_id,
            entity_id_to_link,
            role,
        )
        return link_comp
    except IntegrityError:  # Should be caught by pre-check
        await session.rollback()
        char_name_fetch = await ecs_functions.get_component(
            session, character_concept_entity_id, CharacterConceptComponent
        )
        char_name = char_name_fetch.concept_name if char_name_fetch else "Unknown Character"
        logger.error(
            "Failed to apply character '%s' to Entity %s (role: '%s'). "
            "Likely duplicate application (this should have been caught by pre-check).",
            char_name,
            entity_id_to_link,
            role,
        )
        return None
    except Exception:
        await session.rollback()
        logger.exception("An unexpected error occurred while applying character link.")
        raise


async def remove_character_from_entity(
    session: AsyncSession,
    entity_id_linked: int,
    character_concept_entity_id: int,
    role: str | None = None,  # Role must match to remove a specific link
) -> bool:
    """Remove a specific character link from an entity."""
    stmt = select(EntityCharacterLinkComponent).where(
        EntityCharacterLinkComponent.entity_id == entity_id_linked,
        EntityCharacterLinkComponent.character_concept_entity_id == character_concept_entity_id,
        EntityCharacterLinkComponent.role_in_asset == role,
    )
    result = await session.execute(stmt)
    link_comp_to_delete = result.scalar_one_or_none()

    if link_comp_to_delete:
        # ecs_functions.remove_component expects the component instance
        await ecs_functions.remove_component(session, link_comp_to_delete)  # This will delete the row
        char_name_fetch = await ecs_functions.get_component(
            session, character_concept_entity_id, CharacterConceptComponent
        )
        char_name = char_name_fetch.concept_name if char_name_fetch else "Unknown Character"
        logger.info(
            "Removed character '%s' (Concept ID: %s, Role: '%s') from Entity ID %s.",
            char_name,
            character_concept_entity_id,
            role,
            entity_id_linked,
        )
        return True

    logger.warning(
        "Character link (Concept ID: %s, Role: '%s') not found on Entity ID %s.",
        character_concept_entity_id,
        role,
        entity_id_linked,
    )
    return False


async def get_characters_for_entity(
    session: AsyncSession, entity_id_linked: int
) -> list[tuple[Entity, str | None]]:  # Returns (Character Concept Entity, role_in_asset)
    """Get all characters linked to a specific entity, along with their roles."""
    stmt = select(
        EntityCharacterLinkComponent.character_concept_entity_id, EntityCharacterLinkComponent.role_in_asset
    ).where(EntityCharacterLinkComponent.entity_id == entity_id_linked)
    result = await session.execute(stmt)

    character_links_info: list[tuple[Entity, str | None]] = []
    for char_concept_id, role in result.all():
        char_concept_entity = await get_character_concept_by_id(session, char_concept_id)
        if char_concept_entity:
            character_links_info.append((char_concept_entity, role))
    return character_links_info


async def get_entities_for_character(
    session: AsyncSession,
    character_concept_entity_id: int,
    role_filter: str | None = None,
    filter_by_role_presence: bool | None = None,  # True for any role, False for no role (NULL)
) -> list[Entity]:
    """Get all entities linked to a specific character concept, optionally filtering by role."""
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
    return list(result.scalars().all())
