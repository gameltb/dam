"""Tests for comic book functions."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from dam_test_utils.types import WorldFactory
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.functions import comic_book_functions as cbs
from dam.functions import ecs_functions as ecs_service
from dam.models.conceptual import (
    ComicBookConceptComponent,
    ComicBookVariantComponent,
    PageLink,
)


@pytest_asyncio.fixture
async def db_session(world_factory: WorldFactory) -> AsyncGenerator[AsyncSession, None]:
    """Create a new database session for a test."""
    world: World = await world_factory("test_world", [])
    async with world.get_context(WorldTransaction)() as tx:
        yield tx.session


@pytest.mark.asyncio
async def test_create_comic_book_concept(db_session: AsyncSession) -> None:
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
    retrieved_comp_2 = await ecs_service.get_component(db_session, concept_entity_2.id, ComicBookConceptComponent)
    assert retrieved_comp_2 is not None
    assert retrieved_comp_2.comic_title == "Solo Story"
    assert retrieved_comp_2.series_title == "Solo Story"


@pytest.mark.asyncio
async def test_link_comic_variant_to_concept(db_session: AsyncSession) -> None:
    """Test linking a file entity as a comic variant to a comic concept."""
    concept_entity = await cbs.create_comic_book_concept(db_session, comic_title="Test Comic")
    file_entity = await ecs_service.create_entity(db_session)

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

    assert variant_comp is not None
    assert variant_comp.entity_id == file_entity.id
    assert variant_comp.conceptual_entity_id == concept_entity.id
    assert variant_comp.language == language
    assert variant_comp.format == file_format
    assert variant_comp.variant_description == variant_description
    assert variant_comp.is_primary_variant is True

    retrieved_variant_comp = await ecs_service.get_component(db_session, file_entity.id, ComicBookVariantComponent)
    assert retrieved_variant_comp is not None
    assert retrieved_variant_comp.entity_id == variant_comp.entity_id


@pytest.mark.asyncio
async def test_link_comic_variant_error_cases(db_session: AsyncSession) -> None:
    """Test error cases for linking comic variants."""
    concept_entity = await cbs.create_comic_book_concept(db_session, comic_title="Error Comic")
    file_entity_1 = await ecs_service.create_entity(db_session)

    with pytest.raises(ValueError, match=r"File Entity with ID 999 not found."):
        await cbs.link_comic_variant_to_concept(db_session, concept_entity.id, 999)

    with pytest.raises(ValueError, match=r"ComicBookConcept Entity with ID 998 not found."):
        await cbs.link_comic_variant_to_concept(db_session, 998, file_entity_1.id)

    not_a_concept_entity = await ecs_service.create_entity(db_session)
    with pytest.raises(ValueError, match=f"Entity ID {not_a_concept_entity.id} is not a valid ComicBookConcept."):
        await cbs.link_comic_variant_to_concept(db_session, not_a_concept_entity.id, file_entity_1.id)

    await cbs.link_comic_variant_to_concept(
        db_session, concept_entity.id, file_entity_1.id, language="en", format="CBZ"
    )
    with pytest.raises(
        ValueError,
        match=f"File Entity ID {file_entity_1.id} is already linked as a ComicBookVariant to ComicBookConcept ID {concept_entity.id}",
    ):
        await cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_1.id, language="fr")

    concept_entity_2 = await cbs.create_comic_book_concept(db_session, comic_title="Another Error Comic")
    file_entity_2 = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept_entity_2.id, file_entity_2.id, language="de")
    with pytest.raises(
        ValueError,
        match=f"File Entity ID {file_entity_2.id} is already a ComicBookVariant of a different ComicBookConcept ID {concept_entity_2.id}",
    ):
        await cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_2.id, language="es")


@pytest.mark.asyncio
async def test_get_variants_for_comic_concept(db_session: AsyncSession) -> None:
    """Test retrieving variants for a comic concept."""
    concept_entity = await cbs.create_comic_book_concept(db_session, comic_title="Variant Test Comic")
    file_entity_1 = await ecs_service.create_entity(db_session)
    file_entity_2 = await ecs_service.create_entity(db_session)
    file_entity_3 = await ecs_service.create_entity(db_session)  # Not linked

    await cbs.link_comic_variant_to_concept(
        db_session, concept_entity.id, file_entity_1.id, language="en", format="PDF"
    )
    await cbs.link_comic_variant_to_concept(
        db_session, concept_entity.id, file_entity_2.id, language="jp", format="CBZ", is_primary=True
    )

    variants = await cbs.get_variants_for_comic_concept(db_session, concept_entity.id)
    assert len(variants) == 2
    variant_ids = {v.id for v in variants}
    assert file_entity_1.id in variant_ids
    assert file_entity_2.id in variant_ids
    assert file_entity_3.id not in variant_ids

    # Assuming the service function orders by is_primary desc, then other criteria
    # The actual order might depend on implementation details if not strictly defined
    # For this test, asserting presence and correct count is key.
    # If specific order is guaranteed by service, test for it. Example:
    # primary_first = sorted(variants, key=lambda v: v.get_component(ComicBookVariantComponent).is_primary_variant, reverse=True)
    # assert variants[0].id == file_entity_2.id # if primary is guaranteed first

    # To make it robust, let's find the primary and non-primary
    primary_variant_entity = None
    non_primary_variant_entity = None
    for v_entity in variants:
        v_comp = await ecs_service.get_component(db_session, v_entity.id, ComicBookVariantComponent)
        if v_comp and v_comp.is_primary_variant:
            primary_variant_entity = v_entity
        else:
            non_primary_variant_entity = v_entity

    assert primary_variant_entity is not None
    assert primary_variant_entity.id == file_entity_2.id
    assert non_primary_variant_entity is not None
    assert non_primary_variant_entity.id == file_entity_1.id

    assert await cbs.get_variants_for_comic_concept(db_session, 999) == []

    concept_no_variants = await cbs.create_comic_book_concept(db_session, comic_title="No Variants Here")
    assert await cbs.get_variants_for_comic_concept(db_session, concept_no_variants.id) == []


@pytest.mark.asyncio
async def test_get_comic_concept_for_variant(db_session: AsyncSession) -> None:
    """Test retrieving the comic concept for a given variant file entity."""
    concept_entity = await cbs.create_comic_book_concept(db_session, comic_title="Owner Comic")
    file_entity_variant = await ecs_service.create_entity(db_session)
    file_entity_not_variant = await ecs_service.create_entity(db_session)

    await cbs.link_comic_variant_to_concept(db_session, concept_entity.id, file_entity_variant.id, language="en")

    retrieved_concept = await cbs.get_comic_concept_for_variant(db_session, file_entity_variant.id)
    assert retrieved_concept is not None
    assert retrieved_concept.id == concept_entity.id

    assert await cbs.get_comic_concept_for_variant(db_session, file_entity_not_variant.id) is None
    assert await cbs.get_comic_concept_for_variant(db_session, 999) is None


@pytest.mark.asyncio
async def test_find_comic_book_concepts(db_session: AsyncSession) -> None:
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
async def test_set_primary_comic_variant(db_session: AsyncSession) -> None:
    """Test setting the primary comic variant."""
    concept_entity = await cbs.create_comic_book_concept(db_session, comic_title="Primary Test")
    v1_e = await ecs_service.create_entity(db_session)
    v2_e = await ecs_service.create_entity(db_session)
    v3_e = await ecs_service.create_entity(db_session)  # Belongs to other_concept
    other_concept_entity = await cbs.create_comic_book_concept(db_session, comic_title="Other Primary")

    # IDs for clarity
    concept_id = concept_entity.id
    v1_id = v1_e.id
    v2_id = v2_e.id
    v3_id = v3_e.id  # Belongs to other_concept_id
    other_concept_id = other_concept_entity.id

    vc1 = await cbs.link_comic_variant_to_concept(db_session, concept_id, v1_id, "en", "PDF", is_primary=False)
    vc2 = await cbs.link_comic_variant_to_concept(db_session, concept_id, v2_id, "en", "CBZ", is_primary=False)
    await cbs.link_comic_variant_to_concept(db_session, other_concept_id, v3_id, "de", "PDF", is_primary=True)

    assert await cbs.set_primary_comic_variant(db_session, v1_id, concept_id) is True

    # Refresh components directly from session to check their state
    refreshed_vc1 = await db_session.get(ComicBookVariantComponent, vc1.entity_id)
    refreshed_vc2 = await db_session.get(ComicBookVariantComponent, vc2.entity_id)
    assert refreshed_vc1 is not None
    assert refreshed_vc1.is_primary_variant is True
    assert refreshed_vc2 is not None
    assert refreshed_vc2.is_primary_variant is False

    primary_v_entity = await cbs.get_primary_variant_for_comic_concept(db_session, concept_id)
    assert primary_v_entity is not None
    assert primary_v_entity.id == v1_id

    assert await cbs.set_primary_comic_variant(db_session, v2_id, concept_id) is True

    refreshed_vc1 = await db_session.get(ComicBookVariantComponent, vc1.entity_id)  # Re-get after potential change
    refreshed_vc2 = await db_session.get(ComicBookVariantComponent, vc2.entity_id)  # Re-get
    assert refreshed_vc1 is not None
    assert refreshed_vc1.is_primary_variant is False
    assert refreshed_vc2 is not None
    assert refreshed_vc2.is_primary_variant is True

    # Setting already primary variant should return True and do nothing
    assert await cbs.set_primary_comic_variant(db_session, v2_id, concept_id) is True
    refreshed_vc2 = await db_session.get(ComicBookVariantComponent, vc2.entity_id)
    assert refreshed_vc2 is not None
    assert refreshed_vc2.is_primary_variant is True

    # Error cases
    assert await cbs.set_primary_comic_variant(db_session, v1_id, 999) is False  # Non-existent concept
    assert await cbs.set_primary_comic_variant(db_session, 998, concept_id) is False  # Non-existent variant
    assert (
        await cbs.set_primary_comic_variant(db_session, v3_id, concept_id) is False
    )  # Variant belongs to other concept

    not_concept_e = await ecs_service.create_entity(db_session)  # Entity that is not a concept
    assert await cbs.set_primary_comic_variant(db_session, v1_id, not_concept_e.id) is False


@pytest.mark.asyncio
async def test_get_primary_variant_for_comic_concept(db_session: AsyncSession) -> None:
    """Test getting the primary variant for a comic concept."""
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Get Primary")
    v1 = await ecs_service.create_entity(db_session)
    v2 = await ecs_service.create_entity(db_session)

    await cbs.link_comic_variant_to_concept(db_session, concept.id, v1.id, "en", "PDF")
    await cbs.link_comic_variant_to_concept(db_session, concept.id, v2.id, "jp", "CBZ", is_primary=True)

    primary = await cbs.get_primary_variant_for_comic_concept(db_session, concept.id)
    assert primary is not None
    assert primary.id == v2.id

    concept_no_primary = await cbs.create_comic_book_concept(
        db_session, comic_title="No Primary Comic"
    )  # Corrected title
    v3 = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept_no_primary.id, v3.id, "fr", "ePub")
    assert await cbs.get_primary_variant_for_comic_concept(db_session, concept_no_primary.id) is None
    assert await cbs.get_primary_variant_for_comic_concept(db_session, 999) is None


@pytest.mark.asyncio
async def test_unlink_comic_variant(db_session: AsyncSession) -> None:
    """Test unlinking a comic variant from its concept."""
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Unlink Test")
    variant_e = await ecs_service.create_entity(db_session)

    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_e.id, "en")
    assert await ecs_service.get_component(db_session, variant_e.id, ComicBookVariantComponent) is not None

    assert await cbs.unlink_comic_variant(db_session, variant_e.id) is True
    assert await ecs_service.get_component(db_session, variant_e.id, ComicBookVariantComponent) is None

    assert await cbs.unlink_comic_variant(db_session, variant_e.id) is False  # Already unlinked
    assert await cbs.unlink_comic_variant(db_session, 999) is False  # Non-existent


@pytest.mark.asyncio
async def test_comic_variant_unique_constraints(db_session: AsyncSession) -> None:
    """Test unique constraints on ComicBookVariantComponent."""
    concept1_entity = await cbs.create_comic_book_concept(db_session, comic_title="Constraint Test", issue_number="1")
    file1_entity = await ecs_service.create_entity(db_session)
    file2_entity = await ecs_service.create_entity(db_session)

    # Store IDs for clarity
    concept1_id = concept1_entity.id
    file1_id = file1_entity.id
    file2_id = file2_entity.id

    # Link first variant
    await cbs.link_comic_variant_to_concept(
        db_session, concept1_id, file1_id, language="en", format="PDF", variant_description="Scan A"
    )

    # Attempt to link second variant with the same unique attributes, expecting failure
    with pytest.raises(IntegrityError):
        await cbs.link_comic_variant_to_concept(
            db_session, concept1_id, file2_id, language="en", format="PDF", variant_description="Scan A"
        )


@pytest.mark.asyncio
async def test_primary_variant_logic_on_link_comic(db_session: AsyncSession) -> None:
    """Test primary variant logic when linking comics."""
    concept_entity = await cbs.create_comic_book_concept(db_session, comic_title="Primary Logic")
    f1_entity = await ecs_service.create_entity(db_session)
    f2_entity = await ecs_service.create_entity(db_session)
    f3_entity = await ecs_service.create_entity(db_session)

    concept_id = concept_entity.id
    f1_id = f1_entity.id
    f2_id = f2_entity.id
    f3_id = f3_entity.id

    vc1_comp = await cbs.link_comic_variant_to_concept(db_session, concept_id, f1_id, "en", "PDF", is_primary=True)
    assert vc1_comp.is_primary_variant is True
    primary_variant_entity = await cbs.get_primary_variant_for_comic_concept(db_session, concept_id)
    assert primary_variant_entity is not None
    assert primary_variant_entity.id == f1_id

    vc2_comp = await cbs.link_comic_variant_to_concept(db_session, concept_id, f2_id, "fr", "CBZ", is_primary=True)

    await db_session.refresh(vc1_comp)  # Refresh vc1_comp to get its updated state
    assert vc1_comp.is_primary_variant is False  # Should now be False
    assert vc2_comp.is_primary_variant is True
    primary_variant_entity = await cbs.get_primary_variant_for_comic_concept(db_session, concept_id)
    assert primary_variant_entity is not None
    assert primary_variant_entity.id == f2_id

    vc3_comp = await cbs.link_comic_variant_to_concept(db_session, concept_id, f3_id, "jp", "ZIP", is_primary=False)
    await db_session.refresh(vc2_comp)  # Refresh vc2_comp
    assert vc2_comp.is_primary_variant is True  # Should remain true
    assert vc3_comp.is_primary_variant is False
    primary_variant_entity = await cbs.get_primary_variant_for_comic_concept(db_session, concept_id)
    assert primary_variant_entity is not None
    assert primary_variant_entity.id == f2_id


@pytest.mark.asyncio
async def test_assign_page_to_comic_variant(db_session: AsyncSession) -> None:
    """Test assigning a page to a comic book variant."""
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Paged Comic")
    variant_entity = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_entity.id, "en", "PDF")

    image_entity1 = await ecs_service.create_entity(db_session)
    image_entity2 = await ecs_service.create_entity(db_session)

    # Use IDs for service calls
    variant_entity_id = variant_entity.id
    image_entity1_id = image_entity1.id
    image_entity2_id = image_entity2.id

    page1_comp = await cbs.assign_page_to_comic_variant(db_session, variant_entity_id, image_entity1_id, 1)
    page2_comp = await cbs.assign_page_to_comic_variant(db_session, variant_entity_id, image_entity2_id, 2)

    assert page1_comp is not None
    assert page1_comp.page_number == 1
    assert page1_comp is not None
    assert page1_comp.page_image.id == image_entity1_id
    assert page2_comp is not None
    assert page2_comp.page_number == 2
    assert page2_comp is not None
    assert page2_comp.page_image.id == image_entity2_id

    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_entity_id)
    assert len(pages) == 2
    assert pages[0].id == image_entity1_id
    assert pages[1].id == image_entity2_id

    image_entity3 = await ecs_service.create_entity(db_session)
    image_entity3_id = image_entity3.id

    assert await cbs.assign_page_to_comic_variant(db_session, variant_entity_id, image_entity3_id, 1) is None
    assert await cbs.assign_page_to_comic_variant(db_session, variant_entity_id, image_entity1_id, 3) is None


@pytest.mark.asyncio
async def test_get_ordered_pages_for_comic_variant(db_session: AsyncSession) -> None:
    """Test getting ordered pages for a comic variant."""
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Ordered Pages Comic")
    variant_e = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_e.id, "en", "CBZ")
    variant_e_id = variant_e.id

    img1 = await ecs_service.create_entity(db_session)
    img2 = await ecs_service.create_entity(db_session)
    img3 = await ecs_service.create_entity(db_session)
    img1_id = img1.id
    img2_id = img2.id
    img3_id = img3.id

    await cbs.assign_page_to_comic_variant(db_session, variant_e_id, img2_id, 2)
    await cbs.assign_page_to_comic_variant(db_session, variant_e_id, img3_id, 3)
    await cbs.assign_page_to_comic_variant(db_session, variant_e_id, img1_id, 1)

    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e_id)
    assert len(pages) == 3
    assert pages[0].id == img1_id
    assert pages[1].id == img2_id
    assert pages[2].id == img3_id

    variant_no_pages_e = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_no_pages_e.id, "fr", "EPUB")
    variant_no_pages_id = variant_no_pages_e.id

    pages_empty = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_no_pages_id)
    assert len(pages_empty) == 0


@pytest.mark.asyncio
async def test_remove_page_from_comic_variant(db_session: AsyncSession) -> None:
    """Test removing a page from a comic book variant."""
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Remove Page Comic")
    variant_e = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_e.id)
    variant_e_id = variant_e.id

    img1 = await ecs_service.create_entity(db_session)
    img2 = await ecs_service.create_entity(db_session)
    img1_id = img1.id
    img2_id = img2.id

    p1_link = await cbs.assign_page_to_comic_variant(db_session, variant_e_id, img1_id, 1)
    await cbs.assign_page_to_comic_variant(db_session, variant_e_id, img2_id, 2)

    assert len(await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e_id)) == 2

    assert p1_link is not None
    await cbs.remove_page_from_comic_variant(db_session, variant_e_id, p1_link.page_image.id)

    pages_after_remove = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e_id)
    assert len(pages_after_remove) == 1
    assert pages_after_remove[0].id == img2_id

    non_existent_image_id = 9999
    await cbs.remove_page_from_comic_variant(db_session, variant_e_id, non_existent_image_id)  # Should do nothing
    assert len(await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e_id)) == 1


@pytest.mark.asyncio
async def test_get_comic_variants_containing_image_as_page(db_session: AsyncSession) -> None:
    """Test getting comic variants that contain a specific image as a page."""
    concept1 = await cbs.create_comic_book_concept(db_session, comic_title="CVCIAP C1")
    variant1_1_e = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept1.id, variant1_1_e.id, "en", "PDF")
    variant1_2_e = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept1.id, variant1_2_e.id, "en", "CBZ")

    concept2 = await cbs.create_comic_book_concept(db_session, comic_title="CVCIAP C2")
    variant2_1_e = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept2.id, variant2_1_e.id, "fr", "PDF")

    # IDs
    variant1_1_id = variant1_1_e.id
    variant2_1_id = variant2_1_e.id

    image1 = await ecs_service.create_entity(db_session)
    image2 = await ecs_service.create_entity(db_session)
    image1_id = image1.id
    image2_id = image2.id

    await cbs.assign_page_to_comic_variant(db_session, variant1_1_id, image1_id, 1)
    await cbs.assign_page_to_comic_variant(db_session, variant2_1_id, image1_id, 5)
    await cbs.assign_page_to_comic_variant(db_session, variant1_1_id, image2_id, 2)

    variants_with_image1 = await cbs.get_comic_variants_containing_image_as_page(db_session, image1_id)
    assert len(variants_with_image1) == 2
    found_variant_ids_for_image1 = {item[0].id for item in variants_with_image1}
    assert variant1_1_id in found_variant_ids_for_image1
    assert variant2_1_id in found_variant_ids_for_image1

    variants_with_image2 = await cbs.get_comic_variants_containing_image_as_page(db_session, image2_id)
    assert len(variants_with_image2) == 1
    assert variants_with_image2[0][0].id == variant1_1_id

    image_unused = await ecs_service.create_entity(db_session)
    variants_with_unused_image = await cbs.get_comic_variants_containing_image_as_page(db_session, image_unused.id)
    assert len(variants_with_unused_image) == 0


@pytest.mark.asyncio
async def test_update_page_order_for_comic_variant(db_session: AsyncSession) -> None:
    """Test updating the page order for a comic variant."""
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Update Order Comic")
    variant_e = await ecs_service.create_entity(db_session)
    await cbs.link_comic_variant_to_concept(db_session, concept.id, variant_e.id)
    variant_e_id = variant_e.id

    img1 = await ecs_service.create_entity(db_session)
    img2 = await ecs_service.create_entity(db_session)
    img3 = await ecs_service.create_entity(db_session)
    img1_id = img1.id
    img2_id = img2.id
    img3_id = img3.id

    await cbs.assign_page_to_comic_variant(db_session, variant_e_id, img1_id, 1)
    await cbs.assign_page_to_comic_variant(db_session, variant_e_id, img2_id, 2)
    await cbs.assign_page_to_comic_variant(db_session, variant_e_id, img3_id, 3)

    new_order_ids = [img3_id, img1_id, img2_id]
    await cbs.update_page_order_for_comic_variant(db_session, variant_e_id, new_order_ids)

    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e_id)
    assert len(pages) == 3
    assert pages[0].id == img3_id

    stmt_check = select(PageLink).where(PageLink.owner_entity_id == variant_e_id).order_by(PageLink.page_number)
    res_check = await db_session.execute(stmt_check)
    links_check = res_check.scalars().all()
    # Ensure relationships are loaded if not already. However, direct fetch like this usually loads scalar attributes.
    # For robustness, if these were accessed via an object already in session, refreshing with relationships might be needed.
    # Here, we are fetching new PageLink objects, so their scalar FKs should be populated.
    # If issues arise, explicit loading options on the select statement would be the next step.
    # For now, assuming scalar FKs are loaded.
    assert links_check[0].page_image.id == img3_id
    assert links_check[0].page_number == 1
    assert links_check[1].page_image.id == img1_id
    assert links_check[1].page_number == 2
    assert links_check[2].page_image.id == img2_id
    assert links_check[2].page_number == 3

    shorter_order_ids = [img1_id]
    await cbs.update_page_order_for_comic_variant(db_session, variant_e_id, shorter_order_ids)
    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e_id)
    assert len(pages) == 1
    assert pages[0].id == img1_id

    await cbs.update_page_order_for_comic_variant(db_session, variant_e_id, [])  # Empty list
    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e_id)
    assert len(pages) == 0

    img4 = await ecs_service.create_entity(db_session)
    img4_id = img4.id

    order_with_new_ids = [img2_id, img1_id, img4_id]
    await cbs.update_page_order_for_comic_variant(db_session, variant_e_id, order_with_new_ids)
    pages = await cbs.get_ordered_pages_for_comic_variant(db_session, variant_e_id)
    assert len(pages) == 3
    assert pages[0].id == img2_id
    assert pages[1].id == img1_id
    assert pages[2].id == img4_id

    with pytest.raises(IntegrityError):
        await cbs.update_page_order_for_comic_variant(db_session, variant_e_id, [img1_id, img1_id])
