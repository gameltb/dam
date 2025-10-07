"""Functions for managing comic book concepts, variants, and pages."""

import logging

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession  # Import AsyncSession

from dam.functions import ecs_functions
from dam.models.conceptual import (
    ComicBookConceptComponent,
    ComicBookVariantComponent,
    PageLink,
)
from dam.models.core.entity import Entity

# from dam.models.properties import FilePropertiesComponent # If checking image type

logger = logging.getLogger(__name__)


async def create_comic_book_concept(  # Made async
    session: AsyncSession,  # Use AsyncSession
    comic_title: str,
    series_title: str | None = None,
    issue_number: str | None = None,
    publication_year: int | None = None,
) -> Entity:
    """Create a new comic book concept."""
    if not comic_title:
        raise ValueError("Comic title cannot be empty for a ComicBookConcept.")

    concept_entity = await ecs_functions.create_entity(session)  # Await async call
    comic_concept_comp = ComicBookConceptComponent(
        comic_title=comic_title,
        series_title=series_title or comic_title,
        issue_number=issue_number,
        publication_year=publication_year,
        concept_name=comic_title,
        concept_description=None,
    )
    await ecs_functions.add_component_to_entity(session, concept_entity.id, comic_concept_comp)
    logger.info(
        "Created ComicBookConcept Entity ID %s for title '%s'%s.",
        concept_entity.id,
        comic_title,
        f", issue {issue_number}" if issue_number else "",
    )
    return concept_entity


async def link_comic_variant_to_concept(  # Made async
    session: AsyncSession,  # Use AsyncSession
    comic_concept_entity_id: int,
    file_entity_id: int,
    language: str | None = None,
    format: str | None = None,
    is_primary: bool = False,
    scan_quality: str | None = None,
    variant_description: str | None = None,
) -> ComicBookVariantComponent:
    """Link a file entity as a variant to a comic book concept."""
    file_entity = await ecs_functions.get_entity(session, file_entity_id)  # Await async call
    if not file_entity:
        raise ValueError(f"File Entity with ID {file_entity_id} not found.")
    # await session.refresh(file_entity, attribute_names=['components']) # Removed refresh

    comic_concept_entity = await ecs_functions.get_entity(session, comic_concept_entity_id)  # Await async call
    if not comic_concept_entity:
        raise ValueError(f"ComicBookConcept Entity with ID {comic_concept_entity_id} not found.")
    # await session.refresh(comic_concept_entity, attribute_names=['components']) # Removed refresh

    comic_concept_comp = await ecs_functions.get_component(
        session, comic_concept_entity_id, ComicBookConceptComponent
    )  # Await
    if not comic_concept_comp:
        raise ValueError(f"Entity ID {comic_concept_entity_id} is not a valid ComicBookConcept.")
    await session.refresh(comic_concept_comp)  # Refresh the component

    existing_variant_comp = await ecs_functions.get_component(
        session, file_entity_id, ComicBookVariantComponent
    )  # Await
    if existing_variant_comp:
        if existing_variant_comp.conceptual_entity_id == comic_concept_entity_id:
            raise ValueError(
                f"File Entity ID {file_entity_id} is already linked as a ComicBookVariant to ComicBookConcept ID {comic_concept_entity_id}."
            )
        raise ValueError(
            f"File Entity ID {file_entity_id} is already a ComicBookVariant of a different ComicBookConcept ID {existing_variant_comp.conceptual_entity_id}."
        )

    if is_primary:
        stmt = select(ComicBookVariantComponent).where(
            ComicBookVariantComponent.conceptual_entity_id == comic_concept_entity_id,
            ComicBookVariantComponent.is_primary_variant,
        )
        result_primary = await session.execute(stmt)  # Await
        existing_primary_variant = result_primary.scalars().first()
        if existing_primary_variant and existing_primary_variant.entity_id != file_entity_id:
            logger.info(
                "Demoting existing primary ComicBookVariant (Entity ID %s) for ComicBookConcept ID %s.",
                existing_primary_variant.entity_id,
                comic_concept_entity_id,
            )
            existing_primary_variant.is_primary_variant = False
            session.add(existing_primary_variant)

    cb_variant_comp = ComicBookVariantComponent(
        language=language,
        format=format,
        is_primary_variant=is_primary,
        scan_quality=scan_quality,
        variant_description=variant_description,
    )
    cb_variant_comp.conceptual_asset = comic_concept_entity  # Ensure relationship is set
    await ecs_functions.add_component_to_entity(session, file_entity.id, cb_variant_comp)
    # await session.flush() # Flushing is handled by add_component_to_entity or caller
    logger.info(
        "Linked File Entity ID %s to ComicBookConcept ID %s as variant: lang='%s', format='%s', desc='%s'.",
        file_entity_id,
        comic_concept_entity_id,
        language,
        format,
        variant_description,
    )
    return cb_variant_comp


async def get_variants_for_comic_concept(
    session: AsyncSession, comic_concept_entity_id: int
) -> list[Entity]:  # Made async
    """Get all variants for a comic book concept."""
    comic_concept_entity = await ecs_functions.get_entity(session, comic_concept_entity_id)  # Await
    if not comic_concept_entity or not await ecs_functions.get_component(  # Await
        session, comic_concept_entity_id, ComicBookConceptComponent
    ):
        logger.warning(
            "ComicBookConcept Entity ID %s not found or is not a valid comic concept.", comic_concept_entity_id
        )
        return []

    stmt = (
        select(Entity)
        .join(ComicBookVariantComponent, Entity.id == ComicBookVariantComponent.entity_id)
        .where(ComicBookVariantComponent.conceptual_entity_id == comic_concept_entity_id)
        .order_by(
            ComicBookVariantComponent.is_primary_variant.desc(),
            ComicBookVariantComponent.language,
            ComicBookVariantComponent.format,
        )
    )
    result = await session.execute(stmt)  # Await
    variant_entities = result.scalars().all()
    return list(variant_entities)


async def get_comic_concept_for_variant(session: AsyncSession, file_entity_id: int) -> Entity | None:  # Made async
    """Get the comic book concept for a given variant file entity."""
    cb_variant_comp = await ecs_functions.get_component(session, file_entity_id, ComicBookVariantComponent)  # Await
    if not cb_variant_comp:
        return None
    return await ecs_functions.get_entity(session, cb_variant_comp.conceptual_entity_id)  # Await


async def find_comic_book_concepts(  # Made async
    session: AsyncSession,  # Use AsyncSession
    comic_title_query: str | None = None,
    series_title_query: str | None = None,
    issue_number: str | None = None,
    publication_year: int | None = None,
) -> list[Entity]:
    """Find comic book concepts based on query parameters."""
    stmt = select(Entity).join(ComicBookConceptComponent, Entity.id == ComicBookConceptComponent.entity_id)

    if comic_title_query:
        stmt = stmt.where(ComicBookConceptComponent.comic_title.ilike(f"%{comic_title_query}%"))
    if series_title_query:
        stmt = stmt.where(ComicBookConceptComponent.series_title.ilike(f"%{series_title_query}%"))
    if issue_number:
        stmt = stmt.where(ComicBookConceptComponent.issue_number == issue_number)
    if publication_year:
        stmt = stmt.where(ComicBookConceptComponent.publication_year == publication_year)

    if not any([comic_title_query, series_title_query, issue_number, publication_year]):
        logger.warning("find_comic_book_concepts called without any query parameters, this might return many results.")

    stmt = stmt.distinct()
    result = await session.execute(stmt)  # Await
    concept_entities = result.scalars().all()
    return list(concept_entities)


async def set_primary_comic_variant(
    session: AsyncSession, file_entity_id: int, comic_concept_entity_id: int
) -> bool:  # Made async
    """Set a specific comic variant as the primary one for its concept."""
    concept_comp = await ecs_functions.get_component(
        session, comic_concept_entity_id, ComicBookConceptComponent
    )  # Await
    if not concept_comp:
        logger.error("Entity ID %s is not a ComicBookConcept.", comic_concept_entity_id)
        return False

    target_variant_comp = await ecs_functions.get_component(session, file_entity_id, ComicBookVariantComponent)  # Await
    if not target_variant_comp or target_variant_comp.conceptual_entity_id != comic_concept_entity_id:
        logger.error(
            "File Entity ID %s is not a variant of ComicBookConcept ID %s.",
            file_entity_id,
            comic_concept_entity_id,
        )
        return False

    if target_variant_comp.is_primary_variant:
        logger.info("File Entity ID %s is already the primary comic variant.", file_entity_id)
        return True

    stmt_current_primary = select(ComicBookVariantComponent).where(
        ComicBookVariantComponent.conceptual_entity_id == comic_concept_entity_id,
        ComicBookVariantComponent.is_primary_variant,
    )
    result_current_primary = await session.execute(stmt_current_primary)  # Await
    current_primary_comp = result_current_primary.scalars().first()

    if current_primary_comp:
        logger.info(
            "Demoting current primary ComicBookVariant (Entity ID: %s) for ComicBookConcept ID %s.",
            current_primary_comp.entity_id,
            comic_concept_entity_id,
        )
        current_primary_comp.is_primary_variant = False
        session.add(current_primary_comp)

    target_variant_comp.is_primary_variant = True
    session.add(target_variant_comp)
    logger.info(
        "Set File Entity ID %s as primary variant for ComicBookConcept ID %s.",
        file_entity_id,
        comic_concept_entity_id,
    )
    return True


async def get_primary_variant_for_comic_concept(
    session: AsyncSession, comic_concept_entity_id: int
) -> Entity | None:  # Made async
    """Get the primary variant for a comic book concept."""
    stmt = (
        select(Entity)
        .join(ComicBookVariantComponent, Entity.id == ComicBookVariantComponent.entity_id)
        .where(ComicBookVariantComponent.conceptual_entity_id == comic_concept_entity_id)
        .where(ComicBookVariantComponent.is_primary_variant)
    )
    result = await session.execute(stmt)  # Await
    return result.scalars().first()


async def unlink_comic_variant(session: AsyncSession, file_entity_id: int) -> bool:  # Made async
    """Unlink a comic variant from its concept."""
    cb_variant_comp = await ecs_functions.get_component(session, file_entity_id, ComicBookVariantComponent)  # Await
    if not cb_variant_comp:
        logger.warning("No ComicBookVariantComponent found for File Entity ID %s. Cannot unlink.", file_entity_id)
        return False
    await session.delete(cb_variant_comp)  # Await
    logger.info(
        "Unlinked File Entity ID %s from ComicBookConcept ID %s by deleting ComicBookVariantComponent for entity %s.",
        file_entity_id,
        cb_variant_comp.conceptual_entity_id,
        cb_variant_comp.entity_id,
    )
    return True


# --- Comic Book Page Management ---


async def assign_page_to_comic_variant(  # Made async
    session: AsyncSession,
    comic_variant_entity_id: int,
    page_image_entity_id: int,
    page_number: int,  # Use AsyncSession
) -> PageLink | None:
    """Assign an image entity as a page to a comic book variant."""
    variant_comp = await ecs_functions.get_component(
        session, comic_variant_entity_id, ComicBookVariantComponent
    )  # Await
    if not variant_comp:
        logger.error(
            "Entity ID %s does not have a ComicBookVariantComponent. Cannot assign pages.", comic_variant_entity_id
        )
        return None

    owner_entity = await ecs_functions.get_entity(session, comic_variant_entity_id)  # Await
    if not owner_entity:
        logger.error("Owner entity (comic variant) ID %s not found unexpectedly.", comic_variant_entity_id)
        return None

    page_image_entity = await ecs_functions.get_entity(session, page_image_entity_id)  # Await
    if not page_image_entity:
        logger.error("Page image Entity ID %s not found.", page_image_entity_id)
        return None

    if page_number <= 0:
        logger.error("Page number must be positive.")
        return None

    # Check for existing page link
    stmt = select(PageLink).where(
        PageLink.owner_entity_id == comic_variant_entity_id,
        PageLink.page_number == page_number,
    )
    result = await session.execute(stmt)
    if result.scalar_one_or_none():
        logger.error("Page number %s already exists for comic variant %s", page_number, comic_variant_entity_id)
        return None

    stmt = select(PageLink).where(
        PageLink.owner_entity_id == comic_variant_entity_id,
        PageLink.page_image_entity_id == page_image_entity_id,
    )
    result = await session.execute(stmt)
    if result.scalar_one_or_none():
        logger.error("Image %s already exists in comic variant %s", page_image_entity_id, comic_variant_entity_id)
        return None

    page_link = PageLink(
        owner=owner_entity,
        page_image=page_image_entity,
        page_number=page_number,
    )
    session.add(page_link)
    await session.flush()  # Await
    logger.info(
        "Assigned image Entity ID %s as page %s to comic variant Entity ID %s.",
        page_image_entity_id,
        page_number,
        comic_variant_entity_id,
    )
    return page_link


async def remove_page_from_comic_variant(
    session: AsyncSession, comic_variant_entity_id: int, page_image_entity_id: int
) -> bool:  # Made async
    """Remove a page from a comic variant by the page's image entity ID."""
    if not await ecs_functions.get_component(session, comic_variant_entity_id, ComicBookVariantComponent):  # Await
        logger.warning("Entity ID %s not a valid ComicBookVariant. Cannot remove page.", comic_variant_entity_id)
        return False

    stmt = select(PageLink).where(
        PageLink.owner_entity_id == comic_variant_entity_id, PageLink.page_image_entity_id == page_image_entity_id
    )
    result = await session.execute(stmt)  # Await
    page_link_instance = result.scalar_one_or_none()

    if page_link_instance:
        await session.delete(page_link_instance)  # Await
        logger.info("Removed page image ID %s from comic variant ID %s.", page_image_entity_id, comic_variant_entity_id)
        return True
    logger.warning("Page image ID %s not found for comic variant ID %s.", page_image_entity_id, comic_variant_entity_id)
    return False


async def remove_page_at_number_from_comic_variant(
    session: AsyncSession, comic_variant_entity_id: int, page_number: int
) -> bool:  # Made async
    """Remove a page from a comic variant by its page number."""
    if not await ecs_functions.get_component(session, comic_variant_entity_id, ComicBookVariantComponent):  # Await
        logger.warning("Entity ID %s not a valid ComicBookVariant. Cannot remove page.", comic_variant_entity_id)
        return False

    stmt = select(PageLink).where(
        PageLink.owner_entity_id == comic_variant_entity_id, PageLink.page_number == page_number
    )
    result = await session.execute(stmt)  # Await
    page_link_instance = result.scalar_one_or_none()

    if page_link_instance:
        await session.delete(page_link_instance)  # Await
        logger.info("Removed page at number %s from comic variant ID %s.", page_number, comic_variant_entity_id)
        return True
    logger.warning("Page at number %s not found for comic variant ID %s.", page_number, comic_variant_entity_id)
    return False


async def get_ordered_pages_for_comic_variant(
    session: AsyncSession, comic_variant_entity_id: int
) -> list[Entity]:  # Made async
    """Get all pages for a comic variant, ordered by page number."""
    if not await ecs_functions.get_component(session, comic_variant_entity_id, ComicBookVariantComponent):  # Await
        logger.warning("Entity ID %s not a valid ComicBookVariant. Cannot get pages.", comic_variant_entity_id)
        return []

    stmt = (
        select(Entity)
        .join(PageLink, Entity.id == PageLink.page_image_entity_id)
        .where(PageLink.owner_entity_id == comic_variant_entity_id)
        .order_by(PageLink.page_number)
    )
    result = await session.execute(stmt)  # Await
    page_entities = result.scalars().all()
    return list(page_entities)


async def get_comic_variants_containing_image_as_page(  # Made async
    session: AsyncSession, page_image_entity_id: int
) -> list[tuple[Entity, int]]:
    """Get all comic variants that contain a specific image as a page."""
    stmt = select(PageLink.owner_entity_id, PageLink.page_number).where(
        PageLink.page_image_entity_id == page_image_entity_id
    )
    result_proxy = await session.execute(stmt)  # Await
    results = result_proxy.all()  # Get all rows from result object

    variant_pages_info: list[tuple[Entity, int]] = []
    for owner_id, page_num in results:
        owner_entity = await ecs_functions.get_entity(session, owner_id)  # Await
        if owner_entity and await ecs_functions.get_component(session, owner_id, ComicBookVariantComponent):  # Await
            variant_pages_info.append((owner_entity, page_num))
        else:
            logger.debug(
                "Owner Entity ID %s linked to image ID %s is not a ComicBookVariant. Skipping.",
                owner_id,
                page_image_entity_id,
            )

    return variant_pages_info


async def update_page_order_for_comic_variant(  # Made async
    session: AsyncSession, comic_variant_entity_id: int, ordered_page_image_entity_ids: list[int]
) -> list[PageLink]:
    """Update the page order for a comic variant."""
    if len(ordered_page_image_entity_ids) != len(set(ordered_page_image_entity_ids)):
        raise IntegrityError(
            "Duplicate page image entity IDs provided.",
            params=None,
            orig=Exception("Duplicate page image entity IDs provided."),
        )

    owner_entity = await ecs_functions.get_entity(session, comic_variant_entity_id)  # Await
    if not owner_entity or not await ecs_functions.get_component(
        session, comic_variant_entity_id, ComicBookVariantComponent
    ):  # Await
        raise ValueError(f"Entity ID {comic_variant_entity_id} is not a valid ComicBookVariant.")

    stmt_delete = delete(PageLink).where(PageLink.owner_entity_id == comic_variant_entity_id)
    await session.execute(stmt_delete)  # Await

    new_page_links: list[PageLink] = []
    for i, page_image_id in enumerate(ordered_page_image_entity_ids):
        page_number = i + 1
        page_image_entity = await ecs_functions.get_entity(session, page_image_id)  # Await
        if not page_image_entity:
            logger.warning("Image Entity ID %s for page %s not found. Skipping.", page_image_id, page_number)
            continue

        page_link = PageLink(
            owner=owner_entity,
            page_image=page_image_entity,
            page_number=page_number,
        )
        session.add(page_link)
        new_page_links.append(page_link)

    try:
        await session.flush()  # Await
        logger.info(
            "Successfully updated page order for comic variant ID %s with %s pages.",
            comic_variant_entity_id,
            len(new_page_links),
        )
    except IntegrityError as e:
        await session.rollback()  # Await
        logger.error("Failed to update page order due to integrity error: %s", e)
        raise
    except Exception as e:
        await session.rollback()  # Await
        logger.error("An unexpected error occurred while updating page order: %s", e)
        raise

    return new_page_links
