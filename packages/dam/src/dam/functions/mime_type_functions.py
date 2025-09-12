from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from dam.functions.ecs_functions import add_component_to_entity, get_component
from dam.models.metadata.mime_type_component import MimeTypeComponent


async def set_entity_mime_type(session: AsyncSession, entity_id: int, mime_type: str) -> MimeTypeComponent:
    """
    Sets the mime type for a given entity.
    If the entity already has a mime type, it will be updated.
    Otherwise, a new mime type component will be added.
    """
    component = await get_component(session, entity_id, MimeTypeComponent)
    if component:
        component.value = mime_type
        await session.flush()
    else:
        component = MimeTypeComponent(value=mime_type)
        await add_component_to_entity(session, entity_id, component)
    return component


async def get_entity_mime_type(session: AsyncSession, entity_id: int) -> Optional[str]:
    """
    Gets the mime type for a given entity.
    Returns the mime type string or None if the entity has no mime type.
    """
    component = await get_component(session, entity_id, MimeTypeComponent)
    if component:
        return component.value
    return None
