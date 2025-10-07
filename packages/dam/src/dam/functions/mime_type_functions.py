"""Functions for managing MIME type concepts and their links to entities."""

import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from dam.functions import ecs_functions
from dam.models.conceptual.mime_type_concept_component import MimeTypeConceptComponent
from dam.models.core.entity import Entity
from dam.models.metadata.content_mime_type_component import ContentMimeTypeComponent

logger = logging.getLogger(__name__)


class MimeTypeConceptNotFoundError(Exception):
    """Custom exception for when a MimeTypeConcept is not found."""

    pass


async def get_mime_type_concept_by_name(session: AsyncSession, name: str) -> Entity:
    """Retrieve a MIME type concept entity by its name."""
    stmt = (
        select(Entity)
        .join(MimeTypeConceptComponent, Entity.id == MimeTypeConceptComponent.entity_id)
        .where(MimeTypeConceptComponent.mime_type == name)
    )
    result = await session.execute(stmt)
    entity = result.scalar_one_or_none()
    if entity is None:
        raise MimeTypeConceptNotFoundError(f"Mime type concept '{name}' not found.")
    return entity


async def get_or_create_mime_type_concept(
    session: AsyncSession,
    mime_type: str,
) -> MimeTypeConceptComponent | None:
    """
    Retrieve an existing MimeTypeConceptComponent by name, or create a new one if not found.

    Returns the MimeTypeConceptComponent instance, or None if creation fails.
    """
    clean_mime_type = mime_type.strip().lower()
    if not clean_mime_type:
        logger.warning("Attempted to get or create a mime type with an empty name.")
        return None

    try:
        mime_type_entity = await get_mime_type_concept_by_name(session, clean_mime_type)
        if mime_type_entity:
            return await ecs_functions.get_component(session, mime_type_entity.id, MimeTypeConceptComponent)
    except MimeTypeConceptNotFoundError:
        logger.info("MimeTypeConcept '%s' not found, creating.", clean_mime_type)
        pass

    new_mime_type_entity = await ecs_functions.create_entity(session)
    new_comp = MimeTypeConceptComponent(
        mime_type=clean_mime_type,
        concept_name=clean_mime_type,
        concept_description=f"Mime type concept for {clean_mime_type}",
    )

    try:
        await ecs_functions.add_component_to_entity(session, new_mime_type_entity.id, new_comp)
        logger.info("Created MimeTypeConcept Entity ID %s with name '%s'.", new_mime_type_entity.id, clean_mime_type)
        return await ecs_functions.get_component(session, new_mime_type_entity.id, MimeTypeConceptComponent)
    except IntegrityError:
        await session.rollback()
        logger.warning(
            "Failed to create MimeTypeConcept '%s' due to a likely race condition. Refetching.", clean_mime_type
        )
        try:
            mime_type_entity = await get_mime_type_concept_by_name(session, clean_mime_type)
            return await ecs_functions.get_component(session, mime_type_entity.id, MimeTypeConceptComponent)
        except MimeTypeConceptNotFoundError:
            logger.error("Failed to create and then retrieve MimeTypeConcept '%s'.", clean_mime_type)
            return None


async def set_content_mime_type(
    session: AsyncSession, entity_id: int, mime_type: str
) -> ContentMimeTypeComponent | None:
    """Set the content mime type for a given entity by linking it to a MimeTypeConcept."""
    mime_type_concept = await get_or_create_mime_type_concept(session, mime_type)
    if not mime_type_concept:
        logger.error("Could not get or create mime type concept for '%s'.", mime_type)
        return None

    content_mime_comp = await ecs_functions.get_component(session, entity_id, ContentMimeTypeComponent)
    if content_mime_comp:
        if content_mime_comp.mime_type_concept_id != mime_type_concept.id:
            content_mime_comp.mime_type_concept_id = mime_type_concept.id
            session.add(content_mime_comp)
            await session.flush()
            logger.info("Updated ContentMimeTypeComponent for entity %s to '%s'.", entity_id, mime_type)
    else:
        content_mime_comp = ContentMimeTypeComponent(mime_type_concept_id=mime_type_concept.id)
        await ecs_functions.add_component_to_entity(session, entity_id, content_mime_comp)
        logger.info("Added ContentMimeTypeComponent for entity %s with mime type '%s'.", entity_id, mime_type)

    return content_mime_comp


async def remove_content_mime_type(session: AsyncSession, entity_id: int) -> None:
    """Remove the content mime type for a given entity."""
    component = await ecs_functions.get_component(session, entity_id, ContentMimeTypeComponent)
    if component:
        await ecs_functions.remove_component(session, component)
        logger.info("Removed ContentMimeTypeComponent from entity %s.", entity_id)


async def get_content_mime_type(session: AsyncSession, entity_id: int) -> str | None:
    """
    Get the mime type for a given entity's content.

    Returns the mime type string or None if the entity has no content mime type.
    """
    component = await ecs_functions.get_component(session, entity_id, ContentMimeTypeComponent)
    if component and component.mime_type_concept:
        return component.mime_type_concept.mime_type
    return None
