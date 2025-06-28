import pytest
from sqlalchemy.orm import Session

from dam.models.conceptual import (
    ComicBookVariantComponent,  # For testing scope
    EntityTagLinkComponent,
    TagConceptComponent,
)
from dam.models.core.entity import Entity
from dam.services import comic_book_service as cbs
from dam.services import ecs_service
from dam.services import tag_service as ts

# --- Test Data Fixtures ---


@pytest.fixture
def global_tag_concept(db_session: Session) -> Entity:
    tag_entity = ts.create_tag_concept(
        db_session, tag_name="GlobalTestTag", scope_type="GLOBAL", description="A test tag applicable anywhere."
    )
    db_session.commit()
    assert tag_entity is not None
    return tag_entity


@pytest.fixture
def component_scoped_tag_concept(db_session: Session) -> Entity:
    # Tag that should only apply to entities with ComicBookConceptComponent
    tag_entity = ts.create_tag_concept(
        db_session,
        tag_name="ComicConceptOnlyTag",
        scope_type="COMPONENT_CLASS_REQUIRED",
        scope_detail="ComicBookConceptComponent",
        description="A tag for comic book concepts only.",
    )
    db_session.commit()
    assert tag_entity is not None
    return tag_entity


@pytest.fixture
def local_scoped_tag_concept_for_comic1(db_session: Session, comic_concept1: Entity) -> Entity:
    # Tag local to comic_concept1
    tag_entity = ts.create_tag_concept(
        db_session,
        tag_name="LocalTagForComic1",
        scope_type="CONCEPTUAL_ASSET_LOCAL",
        scope_detail=str(comic_concept1.id),
        description="A local tag for Comic 1 and its variants.",
    )
    db_session.commit()
    assert tag_entity is not None
    return tag_entity


@pytest.fixture
def comic_concept1(db_session: Session) -> Entity:
    concept = cbs.create_comic_book_concept(db_session, comic_title="Test Comic Alpha", issue_number="1")
    db_session.commit()
    assert concept is not None
    return concept


@pytest.fixture
def comic_variant1_of_concept1(db_session: Session, comic_concept1: Entity) -> Entity:
    variant_file_entity = ecs_service.create_entity(db_session)
    cbs.link_comic_variant_to_concept(
        db_session,
        comic_concept_entity_id=comic_concept1.id,
        file_entity_id=variant_file_entity.id,
        language="en",
        format="PDF",
    )
    db_session.commit()
    assert variant_file_entity is not None
    # Add assertion here to check the linked component
    retrieved_variant_comp = ecs_service.get_component(db_session, variant_file_entity.id, ComicBookVariantComponent)
    assert retrieved_variant_comp is not None
    assert retrieved_variant_comp.conceptual_entity_id == comic_concept1.id, (
        f"Variant {variant_file_entity.id} not linked to concept {comic_concept1.id} correctly in fixture"
    )
    return variant_file_entity


@pytest.fixture
def generic_entity1(db_session: Session) -> Entity:
    entity = ecs_service.create_entity(db_session)
    db_session.commit()
    assert entity is not None
    return entity


# --- Tag Definition Tests ---


def test_create_tag_concept(db_session: Session):
    tag_name = "MyUniqueTag"
    tag_entity = ts.create_tag_concept(
        db_session, tag_name=tag_name, scope_type="GLOBAL", description="Test Description", allow_values=True
    )
    db_session.commit()
    assert tag_entity is not None

    comp = ecs_service.get_component(db_session, tag_entity.id, TagConceptComponent)
    assert comp is not None
    assert comp.tag_name == tag_name
    assert comp.tag_scope_type == "GLOBAL"
    assert comp.tag_description == "Test Description"
    assert comp.allow_values is True

    # Test duplicate name
    assert ts.create_tag_concept(db_session, tag_name, "GLOBAL") is tag_entity  # Should return existing

    # Test empty name
    with pytest.raises(ValueError, match="Tag name cannot be empty"):
        ts.create_tag_concept(db_session, "", "GLOBAL")


def test_get_tag_concept_by_name(db_session: Session, global_tag_concept: Entity):
    retrieved_tag = ts.get_tag_concept_by_name(db_session, "GlobalTestTag")
    assert retrieved_tag is not None
    assert retrieved_tag.id == global_tag_concept.id
    assert ts.get_tag_concept_by_name(db_session, "NonExistentTag") is None


def test_get_tag_concept_by_id(db_session: Session, global_tag_concept: Entity):
    retrieved_tag = ts.get_tag_concept_by_id(db_session, global_tag_concept.id)
    assert retrieved_tag is not None
    assert retrieved_tag.id == global_tag_concept.id
    assert ts.get_tag_concept_by_id(db_session, 99999) is None


def test_find_tag_concepts(db_session: Session, global_tag_concept: Entity, component_scoped_tag_concept: Entity):
    all_tags = ts.find_tag_concepts(db_session)
    assert len(all_tags) >= 2  # At least the two created by fixtures

    global_tags = ts.find_tag_concepts(db_session, scope_type="GLOBAL")
    assert global_tag_concept in global_tags
    assert component_scoped_tag_concept not in global_tags

    named_tags = ts.find_tag_concepts(db_session, query_name="GlobalTest")
    assert len(named_tags) == 1
    assert named_tags[0].id == global_tag_concept.id


def test_update_tag_concept(db_session: Session, global_tag_concept: Entity):
    updated_comp = ts.update_tag_concept(
        db_session, global_tag_concept.id, name="GlobalTestTagUpdated", description="New Desc", allow_values=True
    )
    db_session.commit()
    assert updated_comp is not None
    assert updated_comp.tag_name == "GlobalTestTagUpdated"
    assert updated_comp.tag_description == "New Desc"
    assert updated_comp.allow_values is True

    # Test update name conflict
    tag2 = ts.create_tag_concept(db_session, "AnotherTag", "GLOBAL")
    db_session.commit()
    assert ts.update_tag_concept(db_session, global_tag_concept.id, name="AnotherTag") is None  # Should fail

    # Test clear description
    ts.update_tag_concept(db_session, global_tag_concept.id, description="__CLEAR__")
    db_session.commit()
    db_session.refresh(updated_comp)
    assert updated_comp.tag_description is None


def test_delete_tag_concept(db_session: Session, generic_entity1: Entity):
    tag_to_delete_entity = ts.create_tag_concept(db_session, "ToDeleteTag", "GLOBAL")
    db_session.commit()
    assert tag_to_delete_entity is not None
    tag_id = tag_to_delete_entity.id

    # Apply it first
    ts.apply_tag_to_entity(db_session, generic_entity1.id, tag_id)
    db_session.commit()
    assert ecs_service.get_component(db_session, generic_entity1.id, EntityTagLinkComponent) is not None

    assert ts.delete_tag_concept(db_session, tag_id) is True
    db_session.commit()
    assert ts.get_tag_concept_by_id(db_session, tag_id) is None
    # Check if link component was cascade deleted
    assert ecs_service.get_component(db_session, generic_entity1.id, EntityTagLinkComponent) is None


# --- Tag Application Tests ---


def test_apply_and_get_tags_for_entity(db_session: Session, global_tag_concept: Entity, generic_entity1: Entity):
    # Apply global tag
    link1 = ts.apply_tag_to_entity(db_session, generic_entity1.id, global_tag_concept.id)
    db_session.commit()
    assert link1 is not None
    assert link1.tag_concept_entity_id == global_tag_concept.id
    assert link1.entity_id == generic_entity1.id
    assert link1.tag_value is None

    # Apply tag that allows value, with a value
    value_tag_concept = ts.create_tag_concept(db_session, "ValueTag", "GLOBAL", allow_values=True)
    db_session.commit()
    assert value_tag_concept is not None
    link2 = ts.apply_tag_to_entity(db_session, generic_entity1.id, value_tag_concept.id, value="TestValue")
    db_session.commit()
    assert link2 is not None
    assert link2.tag_value == "TestValue"

    # Get tags for entity
    applied_tags = ts.get_tags_for_entity(db_session, generic_entity1.id)
    assert len(applied_tags) == 2
    tag_names_on_entity = {
        ecs_service.get_component(db_session, tag_concept_e.id, TagConceptComponent).tag_name
        for tag_concept_e, val in applied_tags
    }
    assert "GlobalTestTag" in tag_names_on_entity
    assert "ValueTag" in tag_names_on_entity

    # Test duplicate tag application (label)
    assert ts.apply_tag_to_entity(db_session, generic_entity1.id, global_tag_concept.id) is None  # Already applied

    # Test duplicate tag application (with value)
    assert (
        ts.apply_tag_to_entity(db_session, generic_entity1.id, value_tag_concept.id, value="TestValue") is None
    )  # Already applied with this value

    # Test applying value to non-value tag
    # global_tag_concept does not allow values.
    # The service should return None because the pre-check for duplicate label tags will find the existing one.
    link3 = ts.apply_tag_to_entity(db_session, generic_entity1.id, global_tag_concept.id, value="ShouldBeIgnored")
    db_session.commit()
    assert link3 is None  # Expecting None because it's a duplicate application of a label tag.


def test_remove_tag_from_entity(db_session: Session, global_tag_concept: Entity, generic_entity1: Entity):
    ts.apply_tag_to_entity(db_session, generic_entity1.id, global_tag_concept.id)
    db_session.commit()
    assert len(ts.get_tags_for_entity(db_session, generic_entity1.id)) == 1

    assert ts.remove_tag_from_entity(db_session, generic_entity1.id, global_tag_concept.id) is True
    db_session.commit()
    assert len(ts.get_tags_for_entity(db_session, generic_entity1.id)) == 0
    assert ts.remove_tag_from_entity(db_session, generic_entity1.id, global_tag_concept.id) is False  # Already removed


def test_get_entities_for_tag(
    db_session: Session, global_tag_concept: Entity, generic_entity1: Entity, comic_concept1: Entity
):
    ts.apply_tag_to_entity(db_session, generic_entity1.id, global_tag_concept.id)
    ts.apply_tag_to_entity(db_session, comic_concept1.id, global_tag_concept.id)

    value_tag = ts.create_tag_concept(db_session, "Status", "GLOBAL", allow_values=True)
    db_session.commit()
    assert value_tag is not None
    ts.apply_tag_to_entity(db_session, generic_entity1.id, value_tag.id, value="Draft")
    ts.apply_tag_to_entity(db_session, comic_concept1.id, value_tag.id, value="Published")
    db_session.commit()

    entities_with_global_tag = ts.get_entities_for_tag(db_session, global_tag_concept.id)
    assert len(entities_with_global_tag) == 2
    assert generic_entity1 in entities_with_global_tag
    assert comic_concept1 in entities_with_global_tag

    entities_with_status_draft = ts.get_entities_for_tag(db_session, value_tag.id, value_filter="Draft")
    assert len(entities_with_status_draft) == 1
    assert entities_with_status_draft[0].id == generic_entity1.id

    entities_with_status_any_value = ts.get_entities_for_tag(db_session, value_tag.id, filter_by_value_presence=True)
    assert len(entities_with_status_any_value) == 2

    # Create a valueless application of Status tag to a new entity
    entity3 = ecs_service.create_entity(db_session)
    db_session.commit()
    ts.apply_tag_to_entity(db_session, entity3.id, value_tag.id, value=None)  # Valueless application
    db_session.commit()

    entities_with_status_no_value = ts.get_entities_for_tag(db_session, value_tag.id, filter_by_value_presence=False)
    assert len(entities_with_status_no_value) == 1
    assert entities_with_status_no_value[0].id == entity3.id


# --- Scope Validation Tests ---


def test_scope_validation_component_class_required(
    db_session: Session,
    component_scoped_tag_concept: Entity,  # Scope: ComicBookConceptComponent
    comic_concept1: Entity,
    generic_entity1: Entity,
):
    # Should succeed: comic_concept1 has ComicBookConceptComponent
    link_success = ts.apply_tag_to_entity(db_session, comic_concept1.id, component_scoped_tag_concept.id)
    db_session.commit()
    assert link_success is not None

    # Should fail: generic_entity1 does not have ComicBookConceptComponent
    link_fail = ts.apply_tag_to_entity(db_session, generic_entity1.id, component_scoped_tag_concept.id)
    assert link_fail is None


def test_scope_validation_conceptual_asset_local(
    db_session: Session,
    comic_concept1: Entity,
    comic_variant1_of_concept1: Entity,
    local_scoped_tag_concept_for_comic1: Entity,  # Scope: local to comic_concept1
):
    # Tagging the conceptual asset itself should succeed
    link_on_concept = ts.apply_tag_to_entity(db_session, comic_concept1.id, local_scoped_tag_concept_for_comic1.id)
    db_session.commit()
    assert link_on_concept is not None

    # Tagging a variant of that conceptual asset should succeed
    link_on_variant = ts.apply_tag_to_entity(
        db_session, comic_variant1_of_concept1.id, local_scoped_tag_concept_for_comic1.id
    )
    db_session.commit()
    assert link_on_variant is not None

    # Tagging another, unrelated entity should fail
    unrelated_entity = ecs_service.create_entity(db_session)
    db_session.commit()
    link_on_unrelated = ts.apply_tag_to_entity(db_session, unrelated_entity.id, local_scoped_tag_concept_for_comic1.id)
    assert link_on_unrelated is None

    # Tagging another comic concept should fail
    comic_concept2 = cbs.create_comic_book_concept(db_session, comic_title="Unrelated Comic")
    db_session.commit()
    link_on_other_concept = ts.apply_tag_to_entity(
        db_session, comic_concept2.id, local_scoped_tag_concept_for_comic1.id
    )
    assert link_on_other_concept is None


def test_scope_validation_invalid_scope_details(db_session: Session, generic_entity1: Entity):
    # COMPONENT_CLASS_REQUIRED with no detail
    no_detail_comp_tag = ts.create_tag_concept(
        db_session, "NoDetailCompTag", "COMPONENT_CLASS_REQUIRED", scope_detail=None
    )
    db_session.commit()
    assert no_detail_comp_tag is not None
    assert ts.apply_tag_to_entity(db_session, generic_entity1.id, no_detail_comp_tag.id) is None

    # COMPONENT_CLASS_REQUIRED with non-existent component class name
    bad_detail_comp_tag = ts.create_tag_concept(
        db_session, "BadDetailCompTag", "COMPONENT_CLASS_REQUIRED", scope_detail="NonExistentComponent"
    )
    db_session.commit()
    assert bad_detail_comp_tag is not None
    assert ts.apply_tag_to_entity(db_session, generic_entity1.id, bad_detail_comp_tag.id) is None

    # CONCEPTUAL_ASSET_LOCAL with no detail
    no_detail_local_tag = ts.create_tag_concept(
        db_session, "NoDetailLocalTag", "CONCEPTUAL_ASSET_LOCAL", scope_detail=None
    )
    db_session.commit()
    assert no_detail_local_tag is not None
    assert ts.apply_tag_to_entity(db_session, generic_entity1.id, no_detail_local_tag.id) is None

    # CONCEPTUAL_ASSET_LOCAL with non-integer detail
    bad_detail_local_tag = ts.create_tag_concept(
        db_session, "BadDetailLocalTag", "CONCEPTUAL_ASSET_LOCAL", scope_detail="not-an-int"
    )
    db_session.commit()
    assert bad_detail_local_tag is not None
    assert ts.apply_tag_to_entity(db_session, generic_entity1.id, bad_detail_local_tag.id) is None

    # CONCEPTUAL_ASSET_LOCAL where scope_detail ID is not a conceptual asset
    non_concept_owner = ecs_service.create_entity(
        db_session
    )  # This entity does not have a BaseConceptualInfoComponent subclass
    db_session.commit()
    local_to_non_concept_tag = ts.create_tag_concept(
        db_session, "LocalToNonConcept", "CONCEPTUAL_ASSET_LOCAL", scope_detail=str(non_concept_owner.id)
    )
    db_session.commit()
    assert local_to_non_concept_tag is not None
    # Applying this tag to anything should fail scope check because the scope itself is defined against a non-conceptual entity
    assert ts.apply_tag_to_entity(db_session, generic_entity1.id, local_to_non_concept_tag.id) is None


def test_unknown_scope_type(db_session: Session, generic_entity1: Entity):
    unknown_scope_tag = ts.create_tag_concept(db_session, "UnknownScopeTag", "MY_CUSTOM_SCOPE_TYPE")
    db_session.commit()
    assert unknown_scope_tag is not None
    # Default behavior for unknown scope is to deny application
    assert ts.apply_tag_to_entity(db_session, generic_entity1.id, unknown_scope_tag.id) is None
