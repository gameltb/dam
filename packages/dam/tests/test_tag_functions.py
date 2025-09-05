import pytest

# --- Test Data Fixtures ---
import pytest_asyncio  # For async fixtures
from sqlalchemy.ext.asyncio import AsyncSession  # For type hint

from dam.functions import comic_book_functions as cbs
from dam.functions import ecs_functions as ecs_service
from dam.functions import tag_functions as ts
from dam.models.conceptual import (
    ComicBookVariantComponent,  # For testing scope
)
from dam.models.core.entity import Entity

# Updated imports for tag components
from dam.models.tags import (
    EntityTagLinkComponent,
    TagConceptComponent,
)


@pytest_asyncio.fixture  # Mark as async fixture
async def global_tag_concept(db_session: AsyncSession) -> Entity:  # Made async, use AsyncSession
    tag_entity = await ts.create_tag_concept(  # Await
        db_session, tag_name="GlobalTestTag", scope_type="GLOBAL", description="A test tag applicable anywhere."
    )
    await db_session.commit()  # Await
    assert tag_entity is not None
    return tag_entity


@pytest_asyncio.fixture  # Mark as async fixture
async def component_scoped_tag_concept(db_session: AsyncSession) -> Entity:  # Made async
    # Tag that should only apply to entities with ComicBookConceptComponent
    tag_entity = await ts.create_tag_concept(  # Await
        db_session,
        tag_name="ComicConceptOnlyTag",
        scope_type="COMPONENT_CLASS_REQUIRED",
        scope_detail="ComicBookConceptComponent",
        description="A tag for comic book concepts only.",
    )
    await db_session.commit()  # Await
    assert tag_entity is not None
    return tag_entity


@pytest_asyncio.fixture  # Mark as async fixture
async def local_scoped_tag_concept_for_comic1(db_session: AsyncSession, comic_concept1: Entity) -> Entity:  # Made async
    # Tag local to comic_concept1
    # comic_concept1 fixture also needs to be async now
    tag_entity = await ts.create_tag_concept(  # Await
        db_session,
        tag_name="LocalTagForComic1",
        scope_type="CONCEPTUAL_ASSET_LOCAL",
        scope_detail=str(
            comic_concept1.id
        ),  # id access should be fine if comic_concept1 is awaited properly by pytest-asyncio
        description="A local tag for Comic 1 and its variants.",
    )
    await db_session.commit()  # Await
    assert tag_entity is not None
    return tag_entity


@pytest_asyncio.fixture  # Mark as async fixture
async def comic_concept1(db_session: AsyncSession) -> Entity:  # Made async
    concept = await cbs.create_comic_book_concept(db_session, comic_title="Test Comic Alpha", issue_number="1")  # Await
    await db_session.commit()  # Await
    assert concept is not None
    return concept


@pytest_asyncio.fixture  # Mark as async fixture
async def comic_variant1_of_concept1(db_session: AsyncSession, comic_concept1: Entity) -> Entity:  # Made async
    # comic_concept1 fixture is now async
    variant_file_entity = await ecs_service.create_entity(db_session)  # Await
    await cbs.link_comic_variant_to_concept(  # Await
        db_session,
        comic_concept_entity_id=comic_concept1.id,  # id access here
        file_entity_id=variant_file_entity.id,
        language="en",
        format="PDF",
    )
    await db_session.commit()  # Await
    assert variant_file_entity is not None
    # Add assertion here to check the linked component
    retrieved_variant_comp = await ecs_service.get_component(
        db_session, variant_file_entity.id, ComicBookVariantComponent
    )  # Await
    assert retrieved_variant_comp is not None
    assert retrieved_variant_comp.conceptual_entity_id == comic_concept1.id, (
        f"Variant {variant_file_entity.id} not linked to concept {comic_concept1.id} correctly in fixture"
    )
    return variant_file_entity


@pytest_asyncio.fixture  # Mark as async fixture
async def generic_entity1(db_session: AsyncSession) -> Entity:  # Made async
    entity = await ecs_service.create_entity(db_session)  # Await
    await db_session.commit()  # Await
    assert entity is not None
    return entity


# --- Tag Definition Tests ---


@pytest.mark.asyncio
async def test_create_tag_concept(db_session: AsyncSession) -> None:  # Async
    tag_name = "MyUniqueTag"
    tag_entity = await ts.create_tag_concept(  # Await
        db_session, tag_name=tag_name, scope_type="GLOBAL", description="Test Description", allow_values=True
    )
    await db_session.commit()  # Await
    assert tag_entity is not None

    comp = await ecs_service.get_component(db_session, tag_entity.id, TagConceptComponent)  # Await
    assert comp is not None
    assert comp.tag_name == tag_name
    assert comp.tag_scope_type == "GLOBAL"
    assert comp.tag_description == "Test Description"
    assert comp.allow_values is True

    # Test duplicate name
    assert await ts.create_tag_concept(db_session, tag_name, "GLOBAL") is tag_entity  # Should return existing # Await

    # Test empty name
    with pytest.raises(ValueError, match="Tag name cannot be empty"):
        await ts.create_tag_concept(db_session, "", "GLOBAL")  # Await


@pytest.mark.asyncio
async def test_get_tag_concept_by_name(db_session: AsyncSession, global_tag_concept: Entity) -> None:  # Async
    retrieved_tag = await ts.get_tag_concept_by_name(db_session, "GlobalTestTag")  # Await
    assert retrieved_tag is not None
    assert retrieved_tag.id == global_tag_concept.id
    with pytest.raises(ts.TagConceptNotFoundError):  # Expect an exception
        await ts.get_tag_concept_by_name(db_session, "NonExistentTag")  # Await


@pytest.mark.asyncio
async def test_get_tag_concept_by_id(db_session: AsyncSession, global_tag_concept: Entity) -> None:  # Async
    retrieved_tag = await ts.get_tag_concept_by_id(db_session, global_tag_concept.id)  # Await
    assert retrieved_tag is not None
    assert retrieved_tag.id == global_tag_concept.id
    assert await ts.get_tag_concept_by_id(db_session, 99999) is None  # Await


@pytest.mark.asyncio
async def test_find_tag_concepts(
    db_session: AsyncSession, global_tag_concept: Entity, component_scoped_tag_concept: Entity
) -> None:  # Async
    all_tags = await ts.find_tag_concepts(db_session)  # Await
    assert len(all_tags) >= 2  # At least the two created by fixtures

    global_tags = await ts.find_tag_concepts(db_session, scope_type="GLOBAL")  # Await
    assert global_tag_concept in global_tags
    assert component_scoped_tag_concept not in global_tags

    named_tags = await ts.find_tag_concepts(db_session, query_name="GlobalTest")  # Await
    assert len(named_tags) == 1
    assert named_tags[0].id == global_tag_concept.id


@pytest.mark.asyncio
async def test_update_tag_concept(db_session: AsyncSession, global_tag_concept: Entity) -> None:  # Async
    updated_comp = await ts.update_tag_concept(  # Await
        db_session, global_tag_concept.id, name="GlobalTestTagUpdated", description="New Desc", allow_values=True
    )
    await db_session.commit()  # Await
    assert updated_comp is not None
    assert updated_comp.tag_name == "GlobalTestTagUpdated"
    assert updated_comp.tag_description == "New Desc"
    assert updated_comp.allow_values is True

    # Test update name conflict
    tag2 = await ts.create_tag_concept(db_session, "AnotherTag", "GLOBAL")  # Await
    await db_session.commit()  # Await
    assert (
        await ts.update_tag_concept(db_session, global_tag_concept.id, name="AnotherTag") is None
    )  # Should fail # Await

    # Test clear description
    await ts.update_tag_concept(db_session, global_tag_concept.id, description="__CLEAR__")  # Await
    await db_session.commit()  # Await
    await db_session.refresh(updated_comp)  # Await
    assert updated_comp.tag_description is None


@pytest.mark.asyncio
async def test_delete_tag_concept(db_session: AsyncSession, generic_entity1: Entity) -> None:  # Async
    tag_to_delete_entity = await ts.create_tag_concept(db_session, "ToDeleteTag", "GLOBAL")  # Await
    await db_session.commit()  # Await
    assert tag_to_delete_entity is not None
    tag_id = tag_to_delete_entity.id

    # Apply it first
    with db_session.no_autoflush:
        await ts.apply_tag_to_entity(db_session, generic_entity1.id, tag_id)  # Await
    await db_session.commit()  # Await
    assert await ecs_service.get_component(db_session, generic_entity1.id, EntityTagLinkComponent) is not None  # Await

    assert await ts.delete_tag_concept(db_session, tag_id) is True  # Await
    await db_session.commit()  # Await
    assert await ts.get_tag_concept_by_id(db_session, tag_id) is None  # Await
    # Check if link component was cascade deleted
    assert await ecs_service.get_component(db_session, generic_entity1.id, EntityTagLinkComponent) is None  # Await


# --- Tag Application Tests ---


@pytest.mark.asyncio
async def test_apply_and_get_tags_for_entity(
    db_session: AsyncSession, global_tag_concept: Entity, generic_entity1: Entity
) -> None:  # Async
    # Apply global tag
    with db_session.no_autoflush:
        link1 = await ts.apply_tag_to_entity(db_session, generic_entity1.id, global_tag_concept.id)  # Await
    await db_session.commit()  # Await
    assert link1 is not None
    assert link1.tag_concept_entity_id == global_tag_concept.id
    assert link1.entity_id == generic_entity1.id
    assert link1.tag_value is None

    # Apply tag that allows value, with a value
    value_tag_concept = await ts.create_tag_concept(db_session, "ValueTag", "GLOBAL", allow_values=True)  # Await
    await db_session.commit()  # Await
    assert value_tag_concept is not None
    with db_session.no_autoflush:
        link2 = await ts.apply_tag_to_entity(
            db_session, generic_entity1.id, value_tag_concept.id, value="TestValue"
        )  # Await
    await db_session.commit()  # Await
    assert link2 is not None
    assert link2.tag_value == "TestValue"

    # Get tags for entity
    applied_tags = await ts.get_tags_for_entity(db_session, generic_entity1.id)  # Await
    assert len(applied_tags) == 2

    tag_names_on_entity = set()
    for tag_concept_e, val in applied_tags:
        comp = await ecs_service.get_component(db_session, tag_concept_e.id, TagConceptComponent)  # Await
        assert comp is not None
        tag_names_on_entity.add(comp.tag_name)

    assert "GlobalTestTag" in tag_names_on_entity
    assert "ValueTag" in tag_names_on_entity

    # Test duplicate tag application (label)
    with db_session.no_autoflush:
        assert (
            await ts.apply_tag_to_entity(db_session, generic_entity1.id, global_tag_concept.id) is None
        )  # Already applied # Await

    # Test duplicate tag application (with value)
    with db_session.no_autoflush:
        assert (
            await ts.apply_tag_to_entity(db_session, generic_entity1.id, value_tag_concept.id, value="TestValue")
            is None  # Await
        )  # Already applied with this value

    # Test applying value to non-value tag
    # global_tag_concept does not allow values.
    # The service should return None because the pre-check for duplicate label tags will find the existing one.
    with db_session.no_autoflush:
        link3 = await ts.apply_tag_to_entity(
            db_session, generic_entity1.id, global_tag_concept.id, value="ShouldBeIgnored"
        )  # Await
    await db_session.commit()  # Await
    assert link3 is None  # Expecting None because it's a duplicate application of a label tag.


@pytest.mark.asyncio
async def test_remove_tag_from_entity(
    db_session: AsyncSession, global_tag_concept: Entity, generic_entity1: Entity
) -> None:  # Async
    with db_session.no_autoflush:
        await ts.apply_tag_to_entity(db_session, generic_entity1.id, global_tag_concept.id)  # Await
    await db_session.commit()  # Await
    assert len(await ts.get_tags_for_entity(db_session, generic_entity1.id)) == 1  # Await

    assert await ts.remove_tag_from_entity(db_session, generic_entity1.id, global_tag_concept.id) is True  # Await
    await db_session.commit()  # Await
    assert len(await ts.get_tags_for_entity(db_session, generic_entity1.id)) == 0  # Await
    assert (
        await ts.remove_tag_from_entity(db_session, generic_entity1.id, global_tag_concept.id) is False
    )  # Already removed # Await


@pytest.mark.asyncio
async def test_get_entities_for_tag(  # Async
    db_session: AsyncSession, global_tag_concept: Entity, generic_entity1: Entity, comic_concept1: Entity
) -> None:
    with db_session.no_autoflush:
        await ts.apply_tag_to_entity(db_session, generic_entity1.id, global_tag_concept.id)  # Await
        await ts.apply_tag_to_entity(db_session, comic_concept1.id, global_tag_concept.id)  # Await

    value_tag = await ts.create_tag_concept(db_session, "Status", "GLOBAL", allow_values=True)  # Await
    await db_session.commit()  # Await
    assert value_tag is not None
    with db_session.no_autoflush:
        await ts.apply_tag_to_entity(db_session, generic_entity1.id, value_tag.id, value="Draft")  # Await
        await ts.apply_tag_to_entity(db_session, comic_concept1.id, value_tag.id, value="Published")  # Await
    await db_session.commit()  # Await

    entities_with_global_tag = await ts.get_entities_for_tag(db_session, global_tag_concept.id)  # Await
    assert len(entities_with_global_tag) == 2
    assert generic_entity1 in entities_with_global_tag
    assert comic_concept1 in entities_with_global_tag

    entities_with_status_draft = await ts.get_entities_for_tag(db_session, value_tag.id, value_filter="Draft")  # Await
    assert len(entities_with_status_draft) == 1
    assert entities_with_status_draft[0].id == generic_entity1.id

    entities_with_status_any_value = await ts.get_entities_for_tag(
        db_session, value_tag.id, filter_by_value_presence=True
    )  # Await
    assert len(entities_with_status_any_value) == 2

    # Create a valueless application of Status tag to a new entity
    entity3 = await ecs_service.create_entity(db_session)  # Await
    await db_session.commit()  # Await
    with db_session.no_autoflush:
        await ts.apply_tag_to_entity(db_session, entity3.id, value_tag.id, value=None)  # Valueless application # Await
    await db_session.commit()  # Await

    entities_with_status_no_value = await ts.get_entities_for_tag(
        db_session, value_tag.id, filter_by_value_presence=False
    )  # Await
    assert len(entities_with_status_no_value) == 1
    assert entities_with_status_no_value[0].id == entity3.id


# --- Scope Validation Tests ---


@pytest.mark.asyncio
async def test_scope_validation_component_class_required(  # Async
    db_session: AsyncSession,
    component_scoped_tag_concept: Entity,  # Scope: ComicBookConceptComponent
    comic_concept1: Entity,
    generic_entity1: Entity,
) -> None:
    # Should succeed: comic_concept1 has ComicBookConceptComponent
    with db_session.no_autoflush:
        link_success = await ts.apply_tag_to_entity(db_session, comic_concept1.id, component_scoped_tag_concept.id)  # Await
    await db_session.commit()  # Await
    assert link_success is not None

    # Should fail: generic_entity1 does not have ComicBookConceptComponent
    with db_session.no_autoflush:
        link_fail = await ts.apply_tag_to_entity(db_session, generic_entity1.id, component_scoped_tag_concept.id)  # Await
    assert link_fail is None


@pytest.mark.asyncio
async def test_scope_validation_conceptual_asset_local(  # Async
    db_session: AsyncSession,
    comic_concept1: Entity,
    comic_variant1_of_concept1: Entity,
    local_scoped_tag_concept_for_comic1: Entity,  # Scope: local to comic_concept1
) -> None:
    # Tagging the conceptual asset itself should succeed
    with db_session.no_autoflush:
        link_on_concept = await ts.apply_tag_to_entity(
            db_session, comic_concept1.id, local_scoped_tag_concept_for_comic1.id
        )  # Await
    await db_session.commit()  # Await
    assert link_on_concept is not None

    # Tagging a variant of that conceptual asset should succeed
    with db_session.no_autoflush:
        link_on_variant = await ts.apply_tag_to_entity(  # Await
            db_session, comic_variant1_of_concept1.id, local_scoped_tag_concept_for_comic1.id
        )
    await db_session.commit()  # Await
    assert link_on_variant is not None

    # Tagging another, unrelated entity should fail
    unrelated_entity = await ecs_service.create_entity(db_session)  # Await
    await db_session.commit()  # Await
    with db_session.no_autoflush:
        link_on_unrelated = await ts.apply_tag_to_entity(
            db_session, unrelated_entity.id, local_scoped_tag_concept_for_comic1.id
        )  # Await
    assert link_on_unrelated is None

    # Tagging another comic concept should fail
    comic_concept2 = await cbs.create_comic_book_concept(db_session, comic_title="Unrelated Comic")  # Await
    await db_session.commit()  # Await
    with db_session.no_autoflush:
        link_on_other_concept = await ts.apply_tag_to_entity(  # Await
            db_session, comic_concept2.id, local_scoped_tag_concept_for_comic1.id
        )
    assert link_on_other_concept is None


@pytest.mark.asyncio
async def test_scope_validation_invalid_scope_details(db_session: AsyncSession, generic_entity1: Entity) -> None:  # Async
    # COMPONENT_CLASS_REQUIRED with no detail
    no_detail_comp_tag = await ts.create_tag_concept(  # Await
        db_session, "NoDetailCompTag", "COMPONENT_CLASS_REQUIRED", scope_detail=None
    )
    await db_session.commit()  # Await
    assert no_detail_comp_tag is not None
    assert await ts.apply_tag_to_entity(db_session, generic_entity1.id, no_detail_comp_tag.id) is None  # Await

    # COMPONENT_CLASS_REQUIRED with non-existent component class name
    bad_detail_comp_tag = await ts.create_tag_concept(  # Await
        db_session, "BadDetailCompTag", "COMPONENT_CLASS_REQUIRED", scope_detail="NonExistentComponent"
    )
    await db_session.commit()  # Await
    assert bad_detail_comp_tag is not None
    assert await ts.apply_tag_to_entity(db_session, generic_entity1.id, bad_detail_comp_tag.id) is None  # Await

    # CONCEPTUAL_ASSET_LOCAL with no detail
    no_detail_local_tag = await ts.create_tag_concept(  # Await
        db_session, "NoDetailLocalTag", "CONCEPTUAL_ASSET_LOCAL", scope_detail=None
    )
    await db_session.commit()  # Await
    assert no_detail_local_tag is not None
    assert await ts.apply_tag_to_entity(db_session, generic_entity1.id, no_detail_local_tag.id) is None  # Await

    # CONCEPTUAL_ASSET_LOCAL with non-integer detail
    bad_detail_local_tag = await ts.create_tag_concept(  # Await
        db_session, "BadDetailLocalTag", "CONCEPTUAL_ASSET_LOCAL", scope_detail="not-an-int"
    )
    await db_session.commit()  # Await
    assert bad_detail_local_tag is not None
    assert await ts.apply_tag_to_entity(db_session, generic_entity1.id, bad_detail_local_tag.id) is None  # Await

    # CONCEPTUAL_ASSET_LOCAL where scope_detail ID is not a conceptual asset
    non_concept_owner = await ecs_service.create_entity(  # Await
        db_session
    )  # This entity does not have a BaseConceptualInfoComponent subclass
    await db_session.commit()  # Await
    local_to_non_concept_tag = await ts.create_tag_concept(  # Await
        db_session, "LocalToNonConcept", "CONCEPTUAL_ASSET_LOCAL", scope_detail=str(non_concept_owner.id)
    )
    await db_session.commit()  # Await
    assert local_to_non_concept_tag is not None
    # Applying this tag to anything should fail scope check because the scope itself is defined against a non-conceptual entity
    assert await ts.apply_tag_to_entity(db_session, generic_entity1.id, local_to_non_concept_tag.id) is None  # Await


@pytest.mark.asyncio
async def test_unknown_scope_type(db_session: AsyncSession, generic_entity1: Entity) -> None:  # Async
    unknown_scope_tag = await ts.create_tag_concept(db_session, "UnknownScopeTag", "MY_CUSTOM_SCOPE_TYPE")  # Await
    await db_session.commit()  # Await
    assert unknown_scope_tag is not None
    # Default behavior for unknown scope is to deny application
    assert await ts.apply_tag_to_entity(db_session, generic_entity1.id, unknown_scope_tag.id) is None  # Await
