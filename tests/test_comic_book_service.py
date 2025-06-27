import pytest
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from dam.models.core.entity import Entity
from dam.models.conceptual import ComicBookConceptComponent, ComicBookVariantComponent
from dam.services import ecs_service
from dam.services import comic_book_service as cbs # Updated import alias


def test_create_comic_book_concept(db_session: Session):
    """Test creating a new comic book concept."""
    comic_title = "The Adventures of Agent Jules"
    series_title = "Agent Jules Chronicles"
    issue_number = "1"
    publication_year = 2024

    concept_entity = cbs.create_comic_book_concept(
        db_session,
        comic_title=comic_title,
        series_title=series_title,
        issue_number=issue_number,
        publication_year=publication_year,
    )
    db_session.commit()

    assert concept_entity is not None
    assert concept_entity.id is not None

    retrieved_comp = ecs_service.get_component(db_session, concept_entity.id, ComicBookConceptComponent)
    assert retrieved_comp is not None
    assert retrieved_comp.comic_title == comic_title
    assert retrieved_comp.series_title == series_title
    assert retrieved_comp.issue_number == issue_number
    assert retrieved_comp.publication_year == publication_year
    assert retrieved_comp.entity_id == concept_entity.id

    with pytest.raises(ValueError, match="Comic title cannot be empty"):
        cbs.create_comic_book_concept(db_session, comic_title="")

    # Test default series_title if not provided
    concept_entity_2 = cbs.create_comic_book_concept(db_session, comic_title="Solo Story")
    db_session.commit()
    retrieved_comp_2 = ecs_service.get_component(db_session, concept_entity_2.id, ComicBookConceptComponent)
    assert retrieved_comp_2.comic_title == "Solo Story"
    assert retrieved_comp_2.series_title == "Solo Story" # Should default to comic_title


def test_link_comic_variant_to_concept(db_session: Session):
    """Test linking a file entity as a comic variant to a comic concept."""
    concept_entity = cbs.create_comic_book_concept(db_session, comic_title="Test Comic")
    file_entity = ecs_service.create_entity(db_session)
    db_session.commit()

    language = "en"
    file_format = "PDF"
    variant_description = "Digital English PDF"

    variant_comp = cbs.link_comic_variant_to_concept(
        db_session,
        comic_concept_entity_id=concept_entity.id,
        file_entity_id=file_entity.id,
        language=language,
        format=file_format,
        variant_description=variant_description,
        is_primary=True,
    )
    db_session.commit()

    assert variant_comp is not None
    assert variant_comp.entity_id == file_entity.id
    assert variant_comp.conceptual_entity_id == concept_entity.id
    assert variant_comp.language == language
    assert variant_comp.format == file_format
    assert variant_comp.variant_description == variant_description
    assert variant_comp.is_primary_variant is True

    retrieved_variant_comp = ecs_service.get_component(db_session, file_entity.id, ComicBookVariantComponent)
    assert retrieved_variant_comp is not None
    assert retrieved_variant_comp.id == variant_comp.id


def test_link_comic_variant_error_cases(db_session: Session):
    """Test error cases for linking comic variants."""
    concept_entity = cbs.create_comic_book_concept(db_session, comic_title="Error Comic")
    file_entity_1 = ecs_service.create_entity(db_session)
    db_session.commit()

    with pytest.raises(ValueError, match="File Entity with ID 999 not found."):
        cbs.link_comic_variant_to_concept(db_session, concept_entity.id, 999)

    with pytest.raises(ValueError, match="ComicBookConcept Entity with ID 998 not found."):
        cbs.link_comic_variant_to_concept(db_session, 998, file_entity_1.id)

    not_a_concept_entity = ecs_service.create_entity(db_session)
    db_session.commit()
    with pytest.raises(ValueError, match=f"Entity ID {not_a_concept_entity.id} is not a valid ComicBookConcept."):
        cbs.link_comic_variant_to_concept(db_session, not_a_concept_entity.id, file_entity_1.id)

    cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_1.id, language="en", format="CBZ")
    db_session.commit()
    with pytest.raises(ValueError, match=f"File Entity ID {file_entity_1.id} is already linked as a ComicBookVariant to ComicBookConcept ID {concept_entity.id}"):
        cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_1.id, language="fr")

    concept_entity_2 = cbs.create_comic_book_concept(db_session, comic_title="Another Error Comic")
    file_entity_2 = ecs_service.create_entity(db_session)
    db_session.commit()
    cbs.link_comic_variant_to_concept(db_session, concept_entity_2.id, file_entity_2.id, language="de")
    db_session.commit()
    with pytest.raises(ValueError, match=f"File Entity ID {file_entity_2.id} is already a ComicBookVariant of a different ComicBookConcept ID {concept_entity_2.id}"):
         cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_2.id, language="es")


def test_get_variants_for_comic_concept(db_session: Session):
    """Test retrieving variants for a comic concept."""
    concept_entity = cbs.create_comic_book_concept(db_session, comic_title="Variant Test Comic")
    file_entity_1 = ecs_service.create_entity(db_session)
    file_entity_2 = ecs_service.create_entity(db_session)
    file_entity_3 = ecs_service.create_entity(db_session) # Not linked
    db_session.commit()

    cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_1.id, language="en", format="PDF")
    cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_2.id, language="jp", format="CBZ", is_primary=True)
    db_session.commit()

    variants = cbs.get_variants_for_comic_concept(db_session, concept_entity.id)
    assert len(variants) == 2
    variant_ids = {v.id for v in variants}
    assert file_entity_1.id in variant_ids
    assert file_entity_2.id in variant_ids
    assert file_entity_3.id not in variant_ids

    # Check order (primary first)
    assert variants[0].id == file_entity_2.id # Primary
    assert variants[1].id == file_entity_1.id

    assert cbs.get_variants_for_comic_concept(db_session, 999) == []

    concept_no_variants = cbs.create_comic_book_concept(db_session, comic_title="No Variants Here")
    db_session.commit()
    assert cbs.get_variants_for_comic_concept(db_session, concept_no_variants.id) == []


def test_get_comic_concept_for_variant(db_session: Session):
    """Test retrieving the comic concept for a given variant file entity."""
    concept_entity = cbs.create_comic_book_concept(db_session, comic_title="Owner Comic")
    file_entity_variant = ecs_service.create_entity(db_session)
    file_entity_not_variant = ecs_service.create_entity(db_session)
    db_session.commit()

    cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_variant.id, language="en")
    db_session.commit()

    retrieved_concept = cbs.get_comic_concept_for_variant(db_session, file_entity_variant.id)
    assert retrieved_concept is not None
    assert retrieved_concept.id == concept_entity.id

    assert cbs.get_comic_concept_for_variant(db_session, file_entity_not_variant.id) is None
    assert cbs.get_comic_concept_for_variant(db_session, 999) is None


def test_find_comic_book_concepts(db_session: Session):
    """Test finding comic book concepts by criteria."""
    cc1 = cbs.create_comic_book_concept(db_session, comic_title="Amazing Adventures", series_title="Amazing Adventures", issue_number="1", publication_year=1980)
    cc2 = cbs.create_comic_book_concept(db_session, comic_title="Amazing Spider-Man", series_title="Amazing Spider-Man", issue_number="100", publication_year=1971)
    cc3 = cbs.create_comic_book_concept(db_session, comic_title="Spectacular Stories", series_title="Spectacular Stories", issue_number="Annual 1", publication_year=1990)
    db_session.commit()

    results_title = cbs.find_comic_book_concepts(db_session, comic_title_query="Amazing")
    assert len(results_title) == 2
    assert {cc1.id, cc2.id} == {e.id for e in results_title}

    results_series = cbs.find_comic_book_concepts(db_session, series_title_query="Amazing Spider-Man")
    assert len(results_series) == 1
    assert results_series[0].id == cc2.id

    results_issue = cbs.find_comic_book_concepts(db_session, issue_number="1")
    assert len(results_issue) == 1
    assert results_issue[0].id == cc1.id

    results_year = cbs.find_comic_book_concepts(db_session, publication_year=1990)
    assert len(results_year) == 1
    assert results_year[0].id == cc3.id

    results_combo = cbs.find_comic_book_concepts(db_session, comic_title_query="Amazing", publication_year=1971)
    assert len(results_combo) == 1
    assert results_combo[0].id == cc2.id

    assert cbs.find_comic_book_concepts(db_session, comic_title_query="NonExistent") == []
    assert cbs.find_comic_book_concepts(db_session) # Test no criteria (should log warning, return all)
    all_concepts = cbs.find_comic_book_concepts(db_session)
    assert len(all_concepts) == 3


def test_set_primary_comic_variant(db_session: Session):
    concept = cbs.create_comic_book_concept(db_session, comic_title="Primary Test")
    v1_e = ecs_service.create_entity(db_session)
    v2_e = ecs_service.create_entity(db_session)
    v3_e = ecs_service.create_entity(db_session) # Variant of another concept
    other_concept = cbs.create_comic_book_concept(db_session, comic_title="Other Primary")
    db_session.commit()

    vc1 = cbs.link_comic_variant_to_concept(db_session, concept.id, v1_e.id, "en", "PDF", is_primary=False)
    vc2 = cbs.link_comic_variant_to_concept(db_session, concept.id, v2_e.id, "en", "CBZ", is_primary=False)
    cbs.link_comic_variant_to_concept(db_session, other_concept.id, v3_e.id, "de", "PDF", is_primary=True)
    db_session.commit()

    assert cbs.set_primary_comic_variant(db_session, v1_e.id, concept.id) is True
    db_session.commit()
    db_session.refresh(vc1); db_session.refresh(vc2)
    assert vc1.is_primary_variant is True
    assert vc2.is_primary_variant is False
    primary_v = cbs.get_primary_variant_for_comic_concept(db_session, concept.id)
    assert primary_v is not None and primary_v.id == v1_e.id

    assert cbs.set_primary_comic_variant(db_session, v2_e.id, concept.id) is True
    db_session.commit()
    db_session.refresh(vc1); db_session.refresh(vc2)
    assert vc1.is_primary_variant is False
    assert vc2.is_primary_variant is True

    assert cbs.set_primary_comic_variant(db_session, v2_e.id, concept.id) is True # Set current again
    db_session.refresh(vc2)
    assert vc2.is_primary_variant is True

    assert cbs.set_primary_comic_variant(db_session, v1_e.id, 999) is False
    assert cbs.set_primary_comic_variant(db_session, 998, concept.id) is False
    assert cbs.set_primary_comic_variant(db_session, v3_e.id, concept.id) is False

    not_concept_e = ecs_service.create_entity(db_session)
    db_session.commit()
    assert cbs.set_primary_comic_variant(db_session, v1_e.id, not_concept_e.id) is False


def test_get_primary_variant_for_comic_concept(db_session: Session):
    concept = cbs.create_comic_book_concept(db_session, comic_title="Get Primary")
    v1 = ecs_service.create_entity(db_session)
    v2 = ecs_service.create_entity(db_session)
    db_session.commit()

    cbs.link_comic_variant_to_concept(db_session, concept.id, v1.id, "en", "PDF")
    cbs.link_comic_variant_to_concept(db_session, concept.id, v2.id, "jp", "CBZ", is_primary=True)
    db_session.commit()

    primary = cbs.get_primary_variant_for_comic_concept(db_session, concept.id)
    assert primary is not None and primary.id == v2.id

    concept_no_primary = cbs.create_comic_book_concept(db_session, "No Primary Comic")
    v3 = ecs_service.create_entity(db_session)
    db_session.commit()
    cbs.link_comic_variant_to_concept(db_session, concept_no_primary.id, v3.id, "fr", "ePub")
    db_session.commit()
    assert cbs.get_primary_variant_for_comic_concept(db_session, concept_no_primary.id) is None
    assert cbs.get_primary_variant_for_comic_concept(db_session, 999) is None


def test_unlink_comic_variant(db_session: Session):
    concept = cbs.create_comic_book_concept(db_session, comic_title="Unlink Test")
    variant_e = ecs_service.create_entity(db_session)
    db_session.commit()

    cbs.link_comic_variant_to_concept(db_session, concept.id, variant_e.id, "en")
    db_session.commit()
    assert ecs_service.get_component(db_session, variant_e.id, ComicBookVariantComponent) is not None

    assert cbs.unlink_comic_variant(db_session, variant_e.id) is True
    db_session.commit()
    assert ecs_service.get_component(db_session, variant_e.id, ComicBookVariantComponent) is None

    assert cbs.unlink_comic_variant(db_session, variant_e.id) is False # Already unlinked
    assert cbs.unlink_comic_variant(db_session, 999) is False # Non-existent


def test_comic_variant_unique_constraints(db_session: Session):
    """Test unique constraints on ComicBookVariantComponent."""
    concept1 = cbs.create_comic_book_concept(db_session, comic_title="Constraint Test", issue_number="1")
    file1 = ecs_service.create_entity(db_session)
    file2 = ecs_service.create_entity(db_session)
    db_session.commit()

    # Link file1
    cbs.link_comic_variant_to_concept(db_session, concept1.id, file1.id, language="en", format="PDF", variant_description="Scan A")
    db_session.commit()

    # Try to link file2 with exact same lang, format, description to concept1
    with pytest.raises(IntegrityError):
        cbs.link_comic_variant_to_concept(db_session, concept1.id, file2.id, language="en", format="PDF", variant_description="Scan A")
        db_session.commit()
    db_session.rollback()

    # Should be fine with different description
    cbs.link_comic_variant_to_concept(db_session, concept1.id, file2.id, language="en", format="PDF", variant_description="Scan B")
    db_session.commit()


def test_primary_variant_logic_on_link_comic(db_session: Session):
    concept = cbs.create_comic_book_concept(db_session, comic_title="Primary Logic")
    f1 = ecs_service.create_entity(db_session)
    f2 = ecs_service.create_entity(db_session)
    f3 = ecs_service.create_entity(db_session)
    db_session.commit()

    vc1 = cbs.link_comic_variant_to_concept(db_session, concept.id, f1.id, "en", "PDF", is_primary=True)
    db_session.commit()
    assert vc1.is_primary_variant is True

    vc2 = cbs.link_comic_variant_to_concept(db_session, concept.id, f2.id, "fr", "CBZ", is_primary=True)
    db_session.commit()
    db_session.refresh(vc1)
    assert vc1.is_primary_variant is False # Demoted
    assert vc2.is_primary_variant is True

    vc3 = cbs.link_comic_variant_to_concept(db_session, concept.id, f3.id, "de", "ePub", is_primary=False)
    db_session.commit()
    db_session.refresh(vc2)
    assert vc2.is_primary_variant is True # Still primary
    assert vc3.is_primary_variant is False

    primary_variant_entity = cbs.get_primary_variant_for_comic_concept(db_session, concept.id)
    assert primary_variant_entity is not None
    assert primary_variant_entity.id == f2.id
