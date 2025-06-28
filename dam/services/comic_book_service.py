import logging
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError

from dam.models.core.entity import Entity
from dam.models.conceptual import (
    ComicBookConceptComponent,
    ComicBookVariantComponent,
    PageLink,
)
from dam.services import ecs_service
# from dam.models.properties import FilePropertiesComponent # If checking image type

logger = logging.getLogger(__name__)


def create_comic_book_concept(
    session: Session,
    comic_title: str,
    series_title: Optional[str] = None,
    issue_number: Optional[str] = None,
    publication_year: Optional[int] = None,
) -> Entity:
    if not comic_title:
        raise ValueError("Comic title cannot be empty for a ComicBookConcept.")

    concept_entity = ecs_service.create_entity(session)
    comic_concept_comp = ComicBookConceptComponent(
        entity=concept_entity,
        comic_title=comic_title,
        series_title=series_title or comic_title,
        issue_number=issue_number,
        publication_year=publication_year,
    )
    session.add(comic_concept_comp)
    logger.info(
        f"Created ComicBookConcept Entity ID {concept_entity.id} for title '{comic_title}'"
        f"{f', issue {issue_number}' if issue_number else ''}."
    )
    return concept_entity


def link_comic_variant_to_concept(
    session: Session,
    comic_concept_entity_id: int,
    file_entity_id: int,
    language: Optional[str] = None,
    format: Optional[str] = None,
    is_primary: bool = False,
    scan_quality: Optional[str] = None,
    variant_description: Optional[str] = None,
) -> ComicBookVariantComponent:
    file_entity = ecs_service.get_entity(session, file_entity_id)
    if not file_entity:
        raise ValueError(f"File Entity with ID {file_entity_id} not found.")

    comic_concept_entity = ecs_service.get_entity(session, comic_concept_entity_id)
    if not comic_concept_entity:
        raise ValueError(f"ComicBookConcept Entity with ID {comic_concept_entity_id} not found.")

    comic_concept_comp = ecs_service.get_component(session, comic_concept_entity_id, ComicBookConceptComponent)
    if not comic_concept_comp:
        raise ValueError(f"Entity ID {comic_concept_entity_id} is not a valid ComicBookConcept.")

    existing_variant_comp = ecs_service.get_component(session, file_entity_id, ComicBookVariantComponent)
    if existing_variant_comp:
        if existing_variant_comp.conceptual_entity_id == comic_concept_entity_id:
            raise ValueError(
                f"File Entity ID {file_entity_id} is already linked as a ComicBookVariant to ComicBookConcept ID {comic_concept_entity_id}."
            )
        else:
            raise ValueError(
                f"File Entity ID {file_entity_id} is already a ComicBookVariant of a different ComicBookConcept ID {existing_variant_comp.conceptual_entity_id}."
            )

    if is_primary:
        stmt = select(ComicBookVariantComponent).where(
            ComicBookVariantComponent.conceptual_entity_id == comic_concept_entity_id,
            ComicBookVariantComponent.is_primary_variant == True,
        )
        existing_primary_variant = session.execute(stmt).scalars().first()
        if existing_primary_variant and existing_primary_variant.entity_id != file_entity_id:
            logger.info(
                f"Demoting existing primary ComicBookVariant (Entity ID {existing_primary_variant.entity_id}) "
                f"for ComicBookConcept ID {comic_concept_entity_id}."
            )
            existing_primary_variant.is_primary_variant = False
            session.add(existing_primary_variant)

    cb_variant_comp = ComicBookVariantComponent(
        entity=file_entity,
        conceptual_asset=comic_concept_entity,
        language=language,
        format=format,
        is_primary_variant=is_primary,
        scan_quality=scan_quality,
        variant_description=variant_description,
    )
    session.add(cb_variant_comp)
    logger.info(
        f"Linked File Entity ID {file_entity_id} to ComicBookConcept ID {comic_concept_entity_id} "
        f"as variant: lang='{language}', format='{format}', desc='{variant_description}'."
    )
    return cb_variant_comp


def get_variants_for_comic_concept(session: Session, comic_concept_entity_id: int) -> List[Entity]:
    comic_concept_entity = ecs_service.get_entity(session, comic_concept_entity_id)
    if not comic_concept_entity or not ecs_service.get_component(session, comic_concept_entity_id, ComicBookConceptComponent):
        logger.warning(f"ComicBookConcept Entity ID {comic_concept_entity_id} not found or is not a valid comic concept.")
        return []

    stmt = (
        select(Entity)
        .join(ComicBookVariantComponent, Entity.id == ComicBookVariantComponent.entity_id)
        .where(ComicBookVariantComponent.conceptual_entity_id == comic_concept_entity_id)
        .order_by(ComicBookVariantComponent.is_primary_variant.desc(), ComicBookVariantComponent.language, ComicBookVariantComponent.format)
    )
    variant_entities = session.execute(stmt).scalars().all()
    return list(variant_entities)


def get_comic_concept_for_variant(session: Session, file_entity_id: int) -> Optional[Entity]:
    cb_variant_comp = ecs_service.get_component(session, file_entity_id, ComicBookVariantComponent)
    if not cb_variant_comp:
        return None
    return ecs_service.get_entity(session, cb_variant_comp.conceptual_entity_id)


def find_comic_book_concepts(
    session: Session,
    comic_title_query: Optional[str] = None,
    series_title_query: Optional[str] = None,
    issue_number: Optional[str] = None,
    publication_year: Optional[int] = None,
) -> List[Entity]:
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
    concept_entities = session.execute(stmt).scalars().all()
    return list(concept_entities)


def set_primary_comic_variant(session: Session, file_entity_id: int, comic_concept_entity_id: int) -> bool:
    concept_comp = ecs_service.get_component(session, comic_concept_entity_id, ComicBookConceptComponent)
    if not concept_comp:
        logger.error(f"Entity ID {comic_concept_entity_id} is not a ComicBookConcept.")
        return False

    target_variant_comp = ecs_service.get_component(session, file_entity_id, ComicBookVariantComponent)
    if not target_variant_comp or target_variant_comp.conceptual_entity_id != comic_concept_entity_id:
        logger.error(f"File Entity ID {file_entity_id} is not a variant of ComicBookConcept ID {comic_concept_entity_id}.")
        return False

    if target_variant_comp.is_primary_variant:
        logger.info(f"File Entity ID {file_entity_id} is already the primary comic variant.")
        return True

    stmt_current_primary = select(ComicBookVariantComponent).where(
        ComicBookVariantComponent.conceptual_entity_id == comic_concept_entity_id,
        ComicBookVariantComponent.is_primary_variant == True
    )
    current_primary_comp = session.execute(stmt_current_primary).scalars().first()

    if current_primary_comp:
        logger.info(f"Demoting current primary ComicBookVariant (Entity ID: {current_primary_comp.entity_id}) for ComicBookConcept ID {comic_concept_entity_id}.")
        current_primary_comp.is_primary_variant = False
        session.add(current_primary_comp)

    target_variant_comp.is_primary_variant = True
    session.add(target_variant_comp)
    logger.info(f"Set File Entity ID {file_entity_id} as primary variant for ComicBookConcept ID {comic_concept_entity_id}.")
    return True


def get_primary_variant_for_comic_concept(session: Session, comic_concept_entity_id: int) -> Optional[Entity]:
    stmt = (
        select(Entity)
        .join(ComicBookVariantComponent, Entity.id == ComicBookVariantComponent.entity_id)
        .where(ComicBookVariantComponent.conceptual_entity_id == comic_concept_entity_id)
        .where(ComicBookVariantComponent.is_primary_variant == True)
    )
    primary_variant_entity = session.execute(stmt).scalars().first()
    return primary_variant_entity


def unlink_comic_variant(session: Session, file_entity_id: int) -> bool:
    cb_variant_comp = ecs_service.get_component(session, file_entity_id, ComicBookVariantComponent)
    if not cb_variant_comp:
        logger.warning(f"No ComicBookVariantComponent found for File Entity ID {file_entity_id}. Cannot unlink.")
        return False
    session.delete(cb_variant_comp)
    logger.info(
        f"Unlinked File Entity ID {file_entity_id} from ComicBookConcept ID {cb_variant_comp.conceptual_entity_id} "
        f"by deleting ComicBookVariantComponent ID {cb_variant_comp.id}."
    )
    return True


# --- Comic Book Page Management ---

def assign_page_to_comic_variant(
    session: Session,
    comic_variant_entity_id: int,
    page_image_entity_id: int,
    page_number: int
) -> Optional[PageLink]:
    variant_comp = ecs_service.get_component(session, comic_variant_entity_id, ComicBookVariantComponent)
    if not variant_comp:
        logger.error(f"Entity ID {comic_variant_entity_id} does not have a ComicBookVariantComponent. Cannot assign pages.")
        return None

    owner_entity = ecs_service.get_entity(session, comic_variant_entity_id)
    if not owner_entity:
        logger.error(f"Owner entity (comic variant) ID {comic_variant_entity_id} not found unexpectedly.")
        return None

    page_image_entity = ecs_service.get_entity(session, page_image_entity_id)
    if not page_image_entity:
        logger.error(f"Page image Entity ID {page_image_entity_id} not found.")
        return None

    if page_number <= 0:
        logger.error("Page number must be positive.")
        return None

    page_link = PageLink(
        owner=owner_entity,                 # Pass Entity object for relationship
        page_image=page_image_entity,       # Pass Entity object for relationship
        page_number=page_number
    )
    try:
        session.add(page_link)
        session.flush()
        logger.info(f"Assigned image Entity ID {page_image_entity_id} as page {page_number} to comic variant Entity ID {comic_variant_entity_id}.")
        return page_link
    except IntegrityError as e:
        session.rollback()
        logger.error(f"Failed to assign page due to integrity error: {e}")
        return None
    except Exception as e:
        session.rollback()
        logger.error(f"An unexpected error occurred while assigning page: {e}")
        raise


def remove_page_from_comic_variant(
    session: Session,
    comic_variant_entity_id: int,
    page_image_entity_id: int
) -> bool:
    if not ecs_service.get_component(session, comic_variant_entity_id, ComicBookVariantComponent):
        logger.warning(f"Entity ID {comic_variant_entity_id} not a valid ComicBookVariant. Cannot remove page.")
        return False

    stmt = select(PageLink).where(
        PageLink.owner_entity_id == comic_variant_entity_id,
        PageLink.page_image_entity_id == page_image_entity_id
    )
    page_link_instance = session.execute(stmt).scalar_one_or_none()

    if page_link_instance:
        session.delete(page_link_instance)
        logger.info(f"Removed page image ID {page_image_entity_id} from comic variant ID {comic_variant_entity_id}.")
        return True
    logger.warning(f"Page image ID {page_image_entity_id} not found for comic variant ID {comic_variant_entity_id}.")
    return False


def remove_page_at_number_from_comic_variant(
    session: Session,
    comic_variant_entity_id: int,
    page_number: int
) -> bool:
    if not ecs_service.get_component(session, comic_variant_entity_id, ComicBookVariantComponent):
        logger.warning(f"Entity ID {comic_variant_entity_id} not a valid ComicBookVariant. Cannot remove page.")
        return False

    stmt = select(PageLink).where(
        PageLink.owner_entity_id == comic_variant_entity_id,
        PageLink.page_number == page_number
    )
    page_link_instance = session.execute(stmt).scalar_one_or_none()

    if page_link_instance:
        session.delete(page_link_instance)
        logger.info(f"Removed page at number {page_number} from comic variant ID {comic_variant_entity_id}.")
        return True
    logger.warning(f"Page at number {page_number} not found for comic variant ID {comic_variant_entity_id}.")
    return False


def get_ordered_pages_for_comic_variant(
    session: Session,
    comic_variant_entity_id: int
) -> List[Entity]:
    if not ecs_service.get_component(session, comic_variant_entity_id, ComicBookVariantComponent):
        logger.warning(f"Entity ID {comic_variant_entity_id} not a valid ComicBookVariant. Cannot get pages.")
        return []

    stmt = (
        select(Entity)
        .join(PageLink, Entity.id == PageLink.page_image_entity_id)
        .where(PageLink.owner_entity_id == comic_variant_entity_id)
        .order_by(PageLink.page_number)
    )
    page_entities = session.execute(stmt).scalars().all()
    return list(page_entities)


def get_comic_variants_containing_image_as_page(
    session: Session,
    page_image_entity_id: int
) -> List[tuple[Entity, int]]:
    stmt = (
        select(PageLink.owner_entity_id, PageLink.page_number)
        .where(PageLink.page_image_entity_id == page_image_entity_id)
    )
    results = session.execute(stmt).all()

    variant_pages_info = []
    for owner_id, page_num in results:
        owner_entity = ecs_service.get_entity(session, owner_id)
        if owner_entity and ecs_service.get_component(session, owner_id, ComicBookVariantComponent):
            variant_pages_info.append((owner_entity, page_num))
        else:
            logger.debug(f"Owner Entity ID {owner_id} linked to image ID {page_image_entity_id} is not a ComicBookVariant. Skipping.")

    return variant_pages_info


def update_page_order_for_comic_variant(
    session: Session,
    comic_variant_entity_id: int,
    ordered_page_image_entity_ids: List[int]
) -> List[PageLink]:
    owner_entity = ecs_service.get_entity(session, comic_variant_entity_id)
    if not owner_entity or not ecs_service.get_component(session, comic_variant_entity_id, ComicBookVariantComponent):
        raise ValueError(f"Entity ID {comic_variant_entity_id} is not a valid ComicBookVariant.")

    stmt_delete = delete(PageLink).where(PageLink.owner_entity_id == comic_variant_entity_id)
    session.execute(stmt_delete)

    new_page_links = []
    for i, page_image_id in enumerate(ordered_page_image_entity_ids):
        page_number = i + 1
        page_image_entity = ecs_service.get_entity(session, page_image_id)
        if not page_image_entity:
             logger.warning(f"Image Entity ID {page_image_id} for page {page_number} not found. Skipping.")
             continue

        page_link = PageLink(
            owner=owner_entity,                 # Pass Entity object
            page_image=page_image_entity,       # Pass Entity object
            page_number=page_number
        )
        session.add(page_link)
        new_page_links.append(page_link)

    try:
        session.flush()
        logger.info(f"Successfully updated page order for comic variant ID {comic_variant_entity_id} with {len(new_page_links)} pages.")
    except IntegrityError as e:
        session.rollback()
        logger.error(f"Failed to update page order due to integrity error: {e}")
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"An unexpected error occurred while updating page order: {e}")
        raise

    return new_page_links
