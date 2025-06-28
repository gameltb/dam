import pytest
from sqlalchemy import select  # Added import
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dam.models.conceptual import ComicBookConceptComponent, ComicBookVariantComponent, PageLink
from dam.services import comic_book_service as cbs
from dam.services import ecs_service

# from dam.models.properties import ImagePropertiesComponent # If checking image type

from sqlalchemy.ext.asyncio import AsyncSession # For type hinting db_session
from dam.models import Entity # Import Entity globally for the module


@pytest.mark.asyncio
async def test_create_comic_book_concept(db_session: AsyncSession):
    """Test creating a new comic book concept."""
    comic_title = "The Adventures of Agent Jules"
    series_title = "Agent Jules Chronicles"
    issue_number = "1"
    publication_year = 2024

    concept_entity = await cbs.create_comic_book_concept(
        db_session,
        comic_title=comic_title,
        series_title=series_title,
        issue_number=issue_number,
        publication_year=publication_year,
    )
    await db_session.commit()

    assert concept_entity is not None
    assert concept_entity.id is not None

    retrieved_comp = await ecs_service.get_component(db_session, concept_entity.id, ComicBookConceptComponent)
    assert retrieved_comp is not None
    assert retrieved_comp.comic_title == comic_title
    assert retrieved_comp.series_title == series_title
    assert retrieved_comp.issue_number == issue_number
    assert retrieved_comp.publication_year == publication_year
    assert retrieved_comp.entity_id == concept_entity.id

    with pytest.raises(ValueError, match="Comic title cannot be empty"):
        await cbs.create_comic_book_concept(db_session, comic_title="")

    concept_entity_2 = await cbs.create_comic_book_concept(db_session, comic_title="Solo Story")
    await db_session.commit()
    retrieved_comp_2 = await ecs_service.get_component(db_session, concept_entity_2.id, ComicBookConceptComponent)
    assert retrieved_comp_2.comic_title == "Solo Story"
    assert retrieved_comp_2.series_title == "Solo Story"


@pytest.mark.asyncio
async def test_link_comic_variant_to_concept(db_session: AsyncSession):
    """Test linking a file entity as a comic variant to a comic concept."""
    concept_entity = await cbs.create_comic_book_concept(db_session, comic_title="Test Comic")
    file_entity = await ecs_service.create_entity(db_session)
    await db_session.commit()

    language = "en"
    file_format = "PDF"
    variant_description = "Digital English PDF"

    variant_comp = await cbs.link_comic_variant_to_concept(
        db_session,
        comic_concept_entity_id=concept_entity.id,
        file_entity_id=file_entity.id,
        language=language,
        format=file_format,
        variant_description=variant_description,
        is_primary=True,
    )
    await db_session.commit()

    assert variant_comp is not None
    assert variant_comp.entity_id == file_entity.id
    assert variant_comp.conceptual_entity_id == concept_entity.id
    assert variant_comp.language == language
    assert variant_comp.format == file_format
    assert variant_comp.variant_description == variant_description
    assert variant_comp.is_primary_variant is True

    retrieved_variant_comp = await ecs_service.get_component(db_session, file_entity.id, ComicBookVariantComponent)
    assert retrieved_variant_comp is not None
    assert retrieved_variant_comp.id == variant_comp.id


@pytest.mark.asyncio
async def test_link_comic_variant_error_cases(db_session: AsyncSession):
    """Test error cases for linking comic variants."""
    concept_entity = await cbs.create_comic_book_concept(db_session, comic_title="Error Comic")
    file_entity_1 = await ecs_service.create_entity(db_session)
    await db_session.commit()

    with pytest.raises(ValueError, match="File Entity with ID 999 not found."):
        await cbs.link_comic_variant_to_concept(db_session, concept_entity.id, 999)

    with pytest.raises(ValueError, match="ComicBookConcept Entity with ID 998 not found."):
        await cbs.link_comic_variant_to_concept(db_session, 998, file_entity_1.id)

    not_a_concept_entity = await ecs_service.create_entity(db_session)
    await db_session.commit()
    with pytest.raises(ValueError, match=f"Entity ID {not_a_concept_entity.id} is not a valid ComicBookConcept."):
        await cbs.link_comic_variant_to_concept(db_session, not_a_concept_entity.id, file_entity_1.id)

    await cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_1.id, language="en", format="CBZ")
    await db_session.commit()
    with pytest.raises(
        ValueError,
        match=f"File Entity ID {file_entity_1.id} is already linked as a ComicBookVariant to ComicBookConcept ID {concept_entity.id}",
    ):
        await cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_1.id, language="fr")

    concept_entity_2 = await cbs.create_comic_book_concept(db_session, comic_title="Another Error Comic")
    file_entity_2 = await ecs_service.create_entity(db_session)
    await db_session.commit()
    await cbs.link_comic_variant_to_concept(db_session, concept_entity_2.id, file_entity_2.id, language="de")
    await db_session.commit()
    with pytest.raises(
        ValueError,
        match=f"File Entity ID {file_entity_2.id} is already a ComicBookVariant of a different ComicBookConcept ID {concept_entity_2.id}",
    ):
        await cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_2.id, language="es")


@pytest.mark.asyncio
async def test_get_variants_for_comic_concept(db_session: AsyncSession):
    """Test retrieving variants for a comic concept."""
    concept_entity = await cbs.create_comic_book_concept(db_session, comic_title="Variant Test Comic")
    file_entity_1 = await ecs_service.create_entity(db_session)
    file_entity_2 = await ecs_service.create_entity(db_session)
    file_entity_3 = await ecs_service.create_entity(db_session)  # Not linked
    await db_session.commit()

    await cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_1.id, language="en", format="PDF")
    await cbs.link_comic_variant_to_concept(
        db_session, concept_entity.id, file_entity_2.id, language="jp", format="CBZ", is_primary=True
    )
    await db_session.commit()

    variants = await cbs.get_variants_for_comic_concept(db_session, concept_entity.id)
    assert len(variants) == 2
    variant_ids = {v.id for v in variants}
    assert file_entity_1.id in variant_ids
    assert file_entity_2.id in variant_ids
    assert file_entity_3.id not in variant_ids

    assert variants[0].id == file_entity_2.id
    assert variants[1].id == file_entity_1.id

    assert await cbs.get_variants_for_comic_concept(db_session, 999) == []

    concept_no_variants = await cbs.create_comic_book_concept(db_session, comic_title="No Variants Here")
    await db_session.commit()
    assert await cbs.get_variants_for_comic_concept(db_session, concept_no_variants.id) == []


@pytest.mark.asyncio
async def test_get_comic_concept_for_variant(db_session: AsyncSession):
    """Test retrieving the comic concept for a given variant file entity."""
    concept_entity = await cbs.create_comic_book_concept(db_session, comic_title="Owner Comic")
    file_entity_variant = await ecs_service.create_entity(db_session)
    file_entity_not_variant = await ecs_service.create_entity(db_session)
    await db_session.commit()

    await cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_variant.id, language="en")
    await db_session.commit()

    retrieved_concept = await cbs.get_comic_concept_for_variant(db_session, file_entity_variant.id)
    assert retrieved_concept is not None
    assert retrieved_concept.id == concept_entity.id

    assert await cbs.get_comic_concept_for_variant(db_session, file_entity_not_variant.id) is None
    assert await cbs.get_comic_concept_for_variant(db_session, 999) is None


@pytest.mark.asyncio
async def test_find_comic_book_concepts(db_session: AsyncSession):
    """Test finding comic book concepts by criteria."""
    cc1 = await cbs.create_comic_book_concept(
        db_session,
        comic_title="Amazing Adventures",
        series_title="Amazing Adventures",
        issue_number="1",
        publication_year=1980,
    )
    cc2 = await cbs.create_comic_book_concept(
        db_session,
        comic_title="Amazing Spider-Man",
        series_title="Amazing Spider-Man",
        issue_number="100",
        publication_year=1971,
    )
    cc3 = await cbs.create_comic_book_concept(
        db_session,
        comic_title="Spectacular Stories",
        series_title="Spectacular Stories",
        issue_number="Annual 1",
        publication_year=1990,
    )
    await db_session.commit()

    results_title = await cbs.find_comic_book_concepts(db_session, comic_title_query="Amazing")
    assert len(results_title) == 2
    assert {cc1.id, cc2.id} == {e.id for e in results_title}

    results_series = await cbs.find_comic_book_concepts(db_session, series_title_query="Amazing Spider-Man")
    assert len(results_series) == 1
    assert results_series[0].id == cc2.id

    results_issue = await cbs.find_comic_book_concepts(db_session, issue_number="1")
    assert len(results_issue) == 1
    assert results_issue[0].id == cc1.id

    results_year = await cbs.find_comic_book_concepts(db_session, publication_year=1990)
    assert len(results_year) == 1
    assert results_year[0].id == cc3.id

    results_combo = await cbs.find_comic_book_concepts(db_session, comic_title_query="Amazing", publication_year=1971)
    assert len(results_combo) == 1
    assert results_combo[0].id == cc2.id

    assert await cbs.find_comic_book_concepts(db_session, comic_title_query="NonExistent") == []
    all_concepts = await cbs.find_comic_book_concepts(db_session)
    assert len(all_concepts) == 3


@pytest.mark.asyncio
async def test_set_primary_comic_variant(db_session: AsyncSession):
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Primary Test")
    v1_e = await ecs_service.create_entity(db_session)
    v2_e = await ecs_service.create_entity(db_session)
    v3_e = await ecs_service.create_entity(db_session)
    other_concept = await cbs.create_comic_book_concept(db_session, comic_title="Other Primary")
    await db_session.commit()

    vc1 = await cbs.link_comic_variant_to_concept(db_session, concept.id, v1_e.id, "en", "PDF", is_primary=False)
    vc2 = await cbs.link_comic_variant_to_concept(db_session, concept.id, v2_e.id, "en", "CBZ", is_primary=False)
    await cbs.link_comic_variant_to_concept(db_session, other_concept.id, v3_e.id, "de", "PDF", is_primary=True)
    await db_session.commit()

    assert await cbs.set_primary_comic_variant(db_session, v1_e.id, concept.id) is True
    await db_session.commit()
    await db_session.refresh(vc1)
    await db_session.refresh(vc2)
    assert vc1.is_primary_variant is True
    assert vc2.is_primary_variant is False
    primary_v = await cbs.get_primary_variant_for_comic_concept(db_session, concept.id)
    assert primary_v is not None and primary_v.id == v1_e.id

    assert await cbs.set_primary_comic_variant(db_session, v2_e.id, concept.id) is True
    await db_session.commit()
    await db_session.refresh(vc1)
    await db_session.refresh(vc2)
    assert vc1.is_primary_variant is False
    assert vc2.is_primary_variant is True

    assert await cbs.set_primary_comic_variant(db_session, v2_e.id, concept.id) is True
    await db_session.refresh(vc2)
    assert vc2.is_primary_variant is True

    assert await cbs.set_primary_comic_variant(db_session, v1_e.id, 999) is False
    assert await cbs.set_primary_comic_variant(db_session, 998, concept.id) is False
    assert await cbs.set_primary_comic_variant(db_session, v3_e.id, concept.id) is False

    not_concept_e = await ecs_service.create_entity(db_session)
    await db_session.commit()
    assert await cbs.set_primary_comic_variant(db_session, v1_e.id, not_concept_e.id) is False


@pytest.mark.asyncio
async def test_get_primary_variant_for_comic_concept(db_session: AsyncSession):
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Get Primary")
    v1 = await ecs_service.create_entity(db_session)
    v2 = await ecs_service.create_entity(db_session)
    await db_session.commit()

    await cbs.link_comic_variant_to_concept(db_session, concept.id, v1.id, "en", "PDF")
    await cbs.link_comic_variant_to_concept(db_session, concept.id, v2.id, "jp", "CBZ", is_primary=True)
    await db_session.commit()

    primary = await cbs.get_primary_variant_for_comic_concept(db_session, concept.id)
    assert primary is not None and primary.id == v2.id

    concept_no_primary = await cbs.create_comic_book_concept(db_session, "No Primary Comic")
    v3 = await ecs_service.create_entity(db_session)
    await db_session.commit()
    await cbs.link_comic_variant_to_concept(db_session, concept_no_primary.id, v3.id, "fr", "ePub")
    await db_session.commit()
    assert await cbs.get_primary_variant_for_comic_concept(db_session, concept_no_primary.id) is None
    assert await cbs.get_primary_variant_for_comic_concept(db_session, 999) is None


@pytest.mark.asyncio
async def test_unlink_comic_variant(db_session: AsyncSession):
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Unlink Test")
    variant_e = await ecs_service.create_entity(db_session)
    await db_session.commit()

    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_e.id, "en")
    await db_session.commit()
    assert await ecs_service.get_component(db_session, variant_e.id, ComicBookVariantComponent) is not None

    assert await cbs.unlink_comic_variant(db_session, variant_e.id) is True
    await db_session.commit()
    assert await ecs_service.get_component(db_session, variant_e.id, ComicBookVariantComponent) is None

    assert await cbs.unlink_comic_variant(db_session, variant_e.id) is False
    assert await cbs.unlink_comic_variant(db_session, 999) is False


@pytest.mark.asyncio
async def test_comic_variant_unique_constraints(db_session: AsyncSession):
    """Test unique constraints on ComicBookVariantComponent."""
    concept1 = await cbs.create_comic_book_concept(db_session, comic_title="Constraint Test", issue_number="1")
    file1 = await ecs_service.create_entity(db_session)
    file2 = await ecs_service.create_entity(db_session)
    await db_session.commit()

    await cbs.link_comic_variant_to_concept(
        db_session, concept1.id, file1.id, language="en", format="PDF", variant_description="Scan A"
    )
    await db_session.commit()

    with pytest.raises(IntegrityError):
        await cbs.link_comic_variant_to_concept(
            db_session, concept1.id, file2.id, language="en", format="PDF", variant_description="Scan A"
        )
        await db_session.commit()
    await db_session.rollback()

    refreshed_concept1 = await db_session.get(Entity, concept1.id)
    if refreshed_concept1 is None:
        raise RuntimeError("Failed to re-fetch concept1 after rollback, this should not happen.")

    refreshed_file2 = await db_session.get(Entity, file2.id)
    if refreshed_file2 is None:
        raise RuntimeError("Failed to re-fetch file2 after rollback, this should not happen.")

    await cbs.link_comic_variant_to_concept(
        db_session, refreshed_concept1.id, refreshed_file2.id, language="en", format="PDF", variant_description="Scan B"
    )
    await db_session.commit()

    refreshed_file1 = await db_session.get(Entity, file1.id)
    if refreshed_file1 is None:
        raise RuntimeError("Failed to re-fetch file1.")

    await cbs.link_comic_variant_to_concept(
        db_session, refreshed_concept1.id, refreshed_file1.id, language="en", format="PDF", variant_description="Scan C"
    )
    await db_session.commit()

    variants = await cbs.get_variants_for_comic_concept(db_session, refreshed_concept1.id)
    assert len(variants) == 2
    variant_descs = sorted([v.variant_description for v in variants if v.variant_description])
    assert variant_descs == ["Scan B", "Scan C"]


@pytest.mark.asyncio
async def test_primary_variant_logic_on_link_comic(db_session: AsyncSession):
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Primary Logic")
    f1 = await ecs_service.create_entity(db_session)
    f2 = await ecs_service.create_entity(db_session)
    f3 = await ecs_service.create_entity(db_session)
    await db_session.commit()

    vc1 = await cbs.link_comic_variant_to_concept(db_session, concept.id, f1.id, "en", "PDF", is_primary=True)
    await db_session.commit()
    assert vc1.is_primary_variant is True
    primary_variant = await cbs.get_primary_variant_for_comic_concept(db_session, concept.id)
    assert primary_variant is not None
    assert primary_variant.id == vc1.entity_id # Corrected to check entity_id of the variant

    vc2 = await cbs.link_comic_variant_to_concept(db_session, concept.id, f2.id, "fr", "CBZ", is_primary=True)
    await db_session.commit()
    await db_session.refresh(vc1)
    assert vc1.is_primary_variant is False
    assert vc2.is_primary_variant is True
    primary_variant = await cbs.get_primary_variant_for_comic_concept(db_session, concept.id)
    assert primary_variant is not None
    assert primary_variant.id == vc2.entity_id # Corrected

    vc3 = await cbs.link_comic_variant_to_concept(db_session, concept.id, f3.id, "jp", "ZIP", is_primary=False)
    await db_session.commit()
    await db_session.refresh(vc2)
    assert vc2.is_primary_variant is True
    assert vc3.is_primary_variant is False
    primary_variant = await cbs.get_primary_variant_for_comic_concept(db_session, concept.id)
    assert primary_variant is not None
    assert primary_variant.id == vc2.entity_id # Corrected


@pytest.mark.asyncio
async def test_assign_page_to_comic_variant(db_session: AsyncSession):
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Paged Comic")
    variant_entity = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_entity.id, "en", "PDF")

    image_entity1 = await ecs_service.create_entity(db_session)
    image_entity2 = await ecs_service.create_entity(db_session)
    await db_session.commit()

    page1_comp = await cbs.assign_page_to_comic_variant(db_session, variant_entity.id, image_entity1.id, 1)
    page2_comp = await cbs.assign_page_to_comic_variant(db_session, variant_entity.id, image_entity2.id, 2)
    await db_session.commit()

    assert page1_comp.page_number == 1
    assert page1_comp.page_image_entity_id == image_entity1.id
    assert page2_comp.page_number == 2
    assert page2_comp.page_image_entity_id == image_entity2.id

    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_entity.id)
    assert len(pages) == 2
    assert pages[0].id == image_entity1.id # get_ordered_pages returns list of image Entities
    assert pages[1].id == image_entity2.id

    image_entity3 = await ecs_service.create_entity(db_session)
    await db_session.commit()
    # Test assigning same page number (should raise IntegrityError or similar)
    # Service function assign_page_to_comic_variant now returns None on failure / constraint violation
    assert await cbs.assign_page_to_comic_variant(db_session, variant_entity.id, image_entity3.id, 1) is None
    await db_session.rollback()

    # Test assigning same image to different page number
    # This should also be prevented by uq_owner_page_image if active
    assert await cbs.assign_page_to_comic_variant(db_session, variant_entity.id, image_entity1.id, 3) is None
    await db_session.rollback()


@pytest.mark.asyncio
async def test_get_ordered_pages_for_comic_variant(db_session: AsyncSession):
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Ordered Pages Comic")
    variant_e = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_e.id, "en", "CBZ")

    img1 = await ecs_service.create_entity(db_session)
    img2 = await ecs_service.create_entity(db_session)
    img3 = await ecs_service.create_entity(db_session)
    await db_session.commit()

    await cbs.assign_page_to_comic_variant(db_session, variant_e.id, img2.id, 2)
    await cbs.assign_page_to_comic_variant(db_session, variant_e.id, img3.id, 3)
    await cbs.assign_page_to_comic_variant(db_session, variant_e.id, img1.id, 1)
    await db_session.commit()

    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e.id)
    assert len(pages) == 3
    assert pages[0].id == img1.id
    assert pages[1].id == img2.id
    assert pages[2].id == img3.id

    variant_no_pages = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_no_pages.id, "fr", "EPUB")
    await db_session.commit()
    pages_empty = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_no_pages.id)
    assert len(pages_empty) == 0


@pytest.mark.asyncio
async def test_remove_page_from_comic_variant(db_session: AsyncSession):
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Remove Page Comic")
    variant_e = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_e.id)

    img1 = await ecs_service.create_entity(db_session)
    img2 = await ecs_service.create_entity(db_session)
    await db_session.commit()

    p1_link = await cbs.assign_page_to_comic_variant(db_session, variant_e.id, img1.id, 1)
    await cbs.assign_page_to_comic_variant(db_session, variant_e.id, img2.id, 2)
    await db_session.commit()

    assert len(await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e.id)) == 2

    assert p1_link is not None # Ensure p1_link was actually created
    await cbs.remove_page_from_comic_variant(db_session, variant_e.id, p1_link.page_image_entity_id)
    await db_session.commit()

    pages_after_remove = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e.id)
    assert len(pages_after_remove) == 1
    assert pages_after_remove[0].id == img2.id

    non_existent_image_id = 9999
    await cbs.remove_page_from_comic_variant(db_session, variant_e.id, non_existent_image_id)
    await db_session.commit()
    assert len(await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e.id)) == 1


@pytest.mark.asyncio
async def test_get_comic_variants_containing_image_as_page(db_session: AsyncSession):
    concept1 = await cbs.create_comic_book_concept(db_session, comic_title="CVCIAP C1")
    variant1_1_e = await ecs_service.create_entity(db_session)
    variant1_1 = await cbs.link_comic_variant_to_concept(db_session, concept1.id, variant1_1_e.id, "en", "PDF")
    variant1_2_e = await ecs_service.create_entity(db_session)
    variant1_2 = await cbs.link_comic_variant_to_concept(db_session, concept1.id, variant1_2_e.id, "en", "CBZ")

    concept2 = await cbs.create_comic_book_concept(db_session, comic_title="CVCIAP C2")
    variant2_1_e = await ecs_service.create_entity(db_session)
    variant2_1 = await cbs.link_comic_variant_to_concept(db_session, concept2.id, variant2_1_e.id, "fr", "PDF")

    image1 = await ecs_service.create_entity(db_session)
    image2 = await ecs_service.create_entity(db_session)
    await db_session.commit()

    await cbs.assign_page_to_comic_variant(db_session, variant1_1_e.id, image1.id, 1)
    await cbs.assign_page_to_comic_variant(db_session, variant2_1_e.id, image1.id, 5)
    await cbs.assign_page_to_comic_variant(db_session, variant1_1_e.id, image2.id, 2)
    await db_session.commit()

    variants_with_image1 = await cbs.get_comic_variants_containing_image_as_page(db_session, image1.id)
    assert len(variants_with_image1) == 2 # Was 3, but variant1_2 does not have image1
    found_variant_ids_for_image1 = {item[0].id for item in variants_with_image1}
    assert variant1_1_e.id in found_variant_ids_for_image1
    assert variant2_1_e.id in found_variant_ids_for_image1
    # assert variant1_2_e.id in found_variant_ids_for_image1 # This was incorrect, variant1_2 doesn't have image1


    variants_with_image2 = await cbs.get_comic_variants_containing_image_as_page(db_session, image2.id)
    assert len(variants_with_image2) == 1
    assert variants_with_image2[0][0].id == variant1_1_e.id

    image_unused = await ecs_service.create_entity(db_session)
    await db_session.commit()
    variants_with_unused_image = await cbs.get_comic_variants_containing_image_as_page(db_session, image_unused.id)
    assert len(variants_with_unused_image) == 0

@pytest.mark.asyncio
async def test_update_page_order_for_comic_variant(db_session: AsyncSession):
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Update Order Comic")
    variant_e = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_e.id)

    img1 = await ecs_service.create_entity(db_session)
    img2 = await ecs_service.create_entity(db_session)
    img3 = await ecs_service.create_entity(db_session)
    await db_session.commit()

    await cbs.assign_page_to_comic_variant(db_session, variant_e.id, img1.id, 1)
    await cbs.assign_page_to_comic_variant(db_session, variant_e.id, img2.id, 2)
    await cbs.assign_page_to_comic_variant(db_session, variant_e.id, img3.id, 3)
    await db_session.commit()

    new_order = [img3.id, img1.id, img2.id]
    await cbs.update_page_order_for_comic_variant(db_session, variant_e.id, new_order)
    await db_session.commit()

    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e.id)
    assert len(pages) == 3
    assert pages[0].id == img3.id # Page images are entities
    # Check page numbers in PageLink table directly for more precise assertion
    stmt_check = select(PageLink).where(PageLink.owner_entity_id == variant_e.id).order_by(PageLink.page_number)
    res_check = await db_session.execute(stmt_check)
    links_check = res_check.scalars().all()
    assert links_check[0].page_image_entity_id == img3.id and links_check[0].page_number == 1
    assert links_check[1].page_image_entity_id == img1.id and links_check[1].page_number == 2
    assert links_check[2].page_image_entity_id == img2.id and links_check[2].page_number == 3


    shorter_order = [img1.id]
    await cbs.update_page_order_for_comic_variant(db_session, variant_e.id, shorter_order)
    await db_session.commit()
    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e.id)
    assert len(pages) == 1
    assert pages[0].id == img1.id

    await cbs.update_page_order_for_comic_variant(db_session, variant_e.id, [])
    await db_session.commit()
    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e.id)
    assert len(pages) == 0

    img4 = await ecs_service.create_entity(db_session)
    await db_session.commit()
    order_with_new_ids = [img2.id, img1.id, img4.id]
    await cbs.update_page_order_for_comic_variant(db_session, variant_e.id, order_with_new_ids)
    await db_session.commit()
    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e.id)
    assert len(pages) == 3
    assert pages[0].id == img2.id
    assert pages[1].id == img1.id
    assert pages[2].id == img4.id

    with pytest.raises(IntegrityError): # If uq_owner_page_image constraint is active
        await cbs.update_page_order_for_comic_variant(db_session, variant_e.id, [img1.id, img1.id])
    await db_session.rollback()
