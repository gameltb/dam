import logging
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import select, or_

from dam.models.core.entity import Entity
from dam.models.conceptual import (
    ComicBookConceptComponent,
    ComicBookVariantComponent,
    # BaseConceptualInfoComponent, # Not directly used for instantiation here
    # BaseVariantInfoComponent,   # Not directly used for instantiation here
)
from dam.services import ecs_service

logger = logging.getLogger(__name__)


def create_comic_book_concept(
    session: Session,
    comic_title: str,
    series_title: Optional[str] = None,
    issue_number: Optional[str] = None,
    publication_year: Optional[int] = None,
) -> Entity:
    """
    Creates a new Comic Book Concept.

    Args:
        session: The SQLAlchemy session.
        comic_title: The main title of the comic book.
        series_title: The series title, if applicable.
        issue_number: The issue number, if applicable.
        publication_year: The publication year, if applicable.

    Returns:
        The newly created Entity representing the Comic Book Concept.
    """
    if not comic_title:
        raise ValueError("Comic title cannot be empty for a ComicBookConcept.")

    # Optional: Check for existing similar comic book concepts to avoid duplicates.
    # This could be complex depending on how uniqueness is defined (e.g., title + issue + year).
    # For now, we allow creation and rely on user diligence or future de-duplication tools.
    # Example check (can be expanded):
    # stmt_check = select(ComicBookConceptComponent).where(
    #     ComicBookConceptComponent.comic_title == comic_title,
    #     ComicBookConceptComponent.issue_number == issue_number, # Handles None correctly
    #     ComicBookConceptComponent.publication_year == publication_year
    # )
    # existing = session.execute(stmt_check).scalars().first()
    # if existing:
    #     logger.warning(f"ComicBookConcept resembling '{comic_title}' issue '{issue_number}' year '{publication_year}' already exists with Entity ID {existing.entity_id}.")
        # return ecs_service.get_entity(session, existing.entity_id)


    concept_entity = ecs_service.create_entity(session)
    comic_concept_comp = ComicBookConceptComponent(
        entity=concept_entity,
        comic_title=comic_title,
        series_title=series_title or comic_title, # Default series_title to comic_title if not provided
        issue_number=issue_number,
        publication_year=publication_year,
    )
    session.add(comic_concept_comp)
    logger.info(
        f"Created ComicBookConcept Entity ID {concept_entity.id} for title '{comic_title}'"
        f"{f', issue L"{issue_number}"' if issue_number else ''}."
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
    """
    Links a File Entity as a Variant to a Comic Book Concept Entity.
    """
    file_entity = ecs_service.get_entity(session, file_entity_id)
    if not file_entity:
        raise ValueError(f"File Entity with ID {file_entity_id} not found.")

    comic_concept_entity = ecs_service.get_entity(session, comic_concept_entity_id)
    if not comic_concept_entity:
        raise ValueError(f"ComicBookConcept Entity with ID {comic_concept_entity_id} not found.")

    # Check if the comic_concept_entity_id actually has a ComicBookConceptComponent
    comic_concept_comp = ecs_service.get_component(session, comic_concept_entity_id, ComicBookConceptComponent)
    if not comic_concept_comp:
        raise ValueError(f"Entity ID {comic_concept_entity_id} is not a valid ComicBookConcept.")

    # Check if this file_entity is already a comic book variant
    existing_variant_comp = ecs_service.get_component(session, file_entity_id, ComicBookVariantComponent)
    if existing_variant_comp:
        if existing_variant_comp.conceptual_entity_id == comic_concept_entity_id:
            raise ValueError(
                f"File Entity ID {file_entity_id} is already linked as a ComicBookVariant to ComicBookConcept ID {comic_concept_entity_id}."
            )
        else:
            # This case implies the file is a ComicBookVariant of *another* ComicBookConcept.
            # A file should probably only be a variant of one concept of a given type.
            # If it could be a variant of different *types* of concepts, that's a more complex scenario.
            raise ValueError(
                f"File Entity ID {file_entity_id} is already a ComicBookVariant of a different ComicBookConcept ID {existing_variant_comp.conceptual_entity_id}."
            )

    # If is_primary is true, ensure no other variant for this comic concept is primary.
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
    """
    Retrieves all File Entities that are variants of a given ComicBookConcept.
    """
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
    """
    Retrieves the ComicBookConcept Entity that a given File Entity is a variant of.
    """
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
    """
    Finds ComicBookConcept Entities based on query criteria.
    Uses 'AND' logic for provided criteria. Partial matches for titles.
    """
    stmt = select(Entity).join(ComicBookConceptComponent, Entity.id == ComicBookConceptComponent.entity_id)

    if comic_title_query:
        stmt = stmt.where(ComicBookConceptComponent.comic_title.ilike(f"%{comic_title_query}%"))
    if series_title_query:
        stmt = stmt.where(ComicBookConceptComponent.series_title.ilike(f"%{series_title_query}%"))
    if issue_number:
         # Exact match for issue number, or handle as ilike if partial is needed
        stmt = stmt.where(ComicBookConceptComponent.issue_number == issue_number)
    if publication_year:
        stmt = stmt.where(ComicBookConceptComponent.publication_year == publication_year)

    if not any([comic_title_query, series_title_query, issue_number, publication_year]):
        logger.warning("find_comic_book_concepts called without any query parameters, this might return many results.")
        # To prevent returning all, one might require at least one parameter or add a default limit.
        # For now, it will proceed if no criteria are given.

    stmt = stmt.distinct()
    concept_entities = session.execute(stmt).scalars().all()
    return list(concept_entities)


def set_primary_comic_variant(session: Session, file_entity_id: int, comic_concept_entity_id: int) -> bool:
    """
    Sets a specific comic variant as the primary variant for a comic concept.
    """
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
    """
    Retrieves the primary File Entity for a given ComicBookConcept.
    """
    stmt = (
        select(Entity)
        .join(ComicBookVariantComponent, Entity.id == ComicBookVariantComponent.entity_id)
        .where(ComicBookVariantComponent.conceptual_entity_id == comic_concept_entity_id)
        .where(ComicBookVariantComponent.is_primary_variant == True)
    )
    primary_variant_entity = session.execute(stmt).scalars().first()
    return primary_variant_entity


def unlink_comic_variant(session: Session, file_entity_id: int) -> bool:
    """
    Removes the ComicBookVariantComponent from a File Entity, unlinking it.
    The File Entity itself is not deleted.
    """
    cb_variant_comp = ecs_service.get_component(session, file_entity_id, ComicBookVariantComponent)
    if not cb_variant_comp:
        logger.warning(f"No ComicBookVariantComponent found for File Entity ID {file_entity_id}. Cannot unlink.")
        return False

    # ecs_service.remove_component should handle session.delete(component)
    # We need to ensure it's the correct way.
    # remove_component(session, component_instance, flush=True) might be better
    # For now, direct delete and let caller manage session commit/flush.
    session.delete(cb_variant_comp)
    logger.info(
        f"Unlinked File Entity ID {file_entity_id} from ComicBookConcept ID {cb_variant_comp.conceptual_entity_id} "
        f"by deleting ComicBookVariantComponent ID {cb_variant_comp.id}."
    )
    return True

# Potential future functions:
# - update_comic_book_concept(...)
# - update_comic_book_variant(...)
# - merge_comic_book_concepts(...)
