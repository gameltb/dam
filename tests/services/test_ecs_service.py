from unittest.mock import MagicMock  # Moved import to top

import pytest
from sqlalchemy.orm import Session

from dam.models import (
    AudioPropertiesComponent,
    BaseComponent,
    Entity,
    FileLocationComponent,
    FilePropertiesComponent,  # Added
    ImageDimensionsComponent,  # Added
)
from dam.services.ecs_service import (
    add_component_to_entity,
    create_entity,
    delete_entity,
    find_entities_by_component_attribute_value,  # Added
    find_entities_with_components,  # Added
    get_component,
    get_components,
    get_entity,
    remove_component,
)


# Fixture to create a sample entity, relies on db_session from conftest.py
@pytest.fixture
def sample_entity(db_session: Session) -> Entity:
    entity = Entity()
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity)  # Ensure it has its ID etc.
    return entity


def test_add_component_to_entity_success(db_session: Session, sample_entity: Entity):
    """Test successfully adding a new component to an existing entity."""
    # Create a new component instance (e.g., FileLocationComponent)
    # Note: For components requiring entity_id/entity in __init__ (due to kw_only and BaseComponent),
    # we are testing add_component_to_entity which handles this association.
    # So, component_instance is created without entity_id/entity here.
    # If FileLocationComponent's __init__ expects entity_id/entity, this test needs adjustment.
    # Let's assume FileLocationComponent can be instantiated with its own fields first.

    # Check FileLocationComponent's __init__ - it inherits from BaseComponent.
    # BaseComponent's __init__ (due to kw_only=True and field order) expects:
    # entity_id, (entity - if tests require it), and then subclass fields.
    # The add_component_to_entity service is designed to set entity_id and entity.
    # So, we should create the component with its own data.

    # FileLocationComponent specific fields: filepath, storage_type
    # It takes entity_id and entity (from BaseComponent __init__), filepath, storage_type
    # The service function add_component_to_entity is what we're testing.
    # It takes an entity_id and a component_instance.
    # The component_instance here is *before* being associated.
    # Its __init__ would have been called like:
    # FileLocationComponent(entity_id=X, entity=Y, filepath="...", storage_type="...")
    # This is tricky. The service is meant to take a *new* component object.

    # Let's assume the component_instance is created with its own data first.
    # The service will then set entity_id and entity.
    # This implies that FileLocationComponent can be initialized without entity_id/entity,
    # which is true for its *own* fields, but BaseComponent init args are inherited.

    # If kw_only=True is fully in effect, FileLocationComponent needs:
    # filepath (required), storage_type (default="local")
    # AND from BaseComponent: entity_id (required), entity (required by tests)
    # This means add_component_to_entity's `component_instance` argument
    # must already have entity_id and entity set if it's to be fully initialized.
    # This makes `add_component_to_entity` slightly less generic if component must be pre-associated.

    # Let's redefine: `add_component_to_entity` takes a *partially* created component
    # (with its own fields set) and completes its association.

    # Instantiate FileLocationComponent with only its own required fields:
    # This will fail if kw_only=True makes entity_id a required init param for FileLocationComponent
    # which it does via BaseComponent.
    # So, the component_instance passed to add_component_to_entity should be created
    # with dummy/None for entity_id/entity if the service is meant to set them.
    # Or, the service is for components already knowing their entity_id.
    # The current service signature `add_component_to_entity(session, entity_id, component_instance)`
    # implies `component_instance` is new and the service links it.

    # For this test, let's create a FileLocationComponent instance as it would be
    # before being passed to the service. Since its __init__ is strict due to kw_only,
    # we must satisfy it, perhaps with placeholder/None for entity-related fields if the
    # service is meant to overwrite them.
    # However, the service sets `component_instance.entity_id` and `component_instance.entity`.
    # This means the component should be creatable without them.
    # The current setup with BaseComponent kw_only=True and entity_id as a required param
    # means this is not possible.

    # Resolution: The `add_component_to_entity` service is for instances that are *already*
    # fully valid dataclass instances. If they need `entity_id` in `__init__`, it must be provided.
    # The service then primarily adds to session and flushes.
    # This makes the `entity_id` arg to the service somewhat redundant if component already has it.

    # Let's assume `add_component_to_entity` is more about linking a NEW component
    # that has its specific data, to an entity.
    # If kw_only=True makes entity_id mandatory for FileLocationComponent init,
    # then we must provide it. The service will then use the passed entity_id to fetch Entity.

    # For a component that doesn't require entity_id in its direct constructor args yet:
    # (This relies on BaseComponent not forcing entity_id into __init__ of a bare component)
    # This is where the kw_only=True inheritance is tricky.
    # Let's assume the service's role is to correctly set entity_id and entity relation.
    # The component instance should be created with its own specific fields.

    # FileLocationComponent fields: filepath (required), storage_type (default 'local')
    # BaseComponent fields (init=True): entity_id, entity (from tests)
    # So, FileLocationComponent.__init__ needs entity_id, entity, filepath, [storage_type]

    # This test will assume that `add_component_to_entity` is given a component
    # that has *not yet* been associated with an entity_id or entity object.
    # This means the component's __init__ must allow this.
    # Our current models with kw_only=True and inherited entity_id from BaseComponent
    # make entity_id a required keyword argument for FileLocationComponent.
    # This makes the premise of `add_component_to_entity` as designed difficult.

    # Re-think: `add_component_to_entity` should take component *data* (a dict or kwargs)
    # and the component *type*, then construct it. This is more robust.
    # For now, stick to current signature and assume component can be made.
    # The test will need to provide the required args for FileLocationComponent.

    # This test is now more about the service correctly adding a pre-constructed component.
    # (The component_instance is created with entity_id and entity because its __init__ requires it)
    component_to_add = FileLocationComponent(
        entity_id=sample_entity.id,
        entity=sample_entity,
        content_identifier="test_content_id_123",  # Changed
        physical_path_or_key="physical/path/to/file.txt",  # Added
        contextual_filename="file.txt",  # Changed
        storage_type="test_local",
    )

    # The service will re-set entity_id and entity, which is fine.
    added_component = add_component_to_entity(db_session, sample_entity.id, component_to_add)
    db_session.commit()  # Commit after service call

    assert added_component is not None
    assert added_component.id is not None  # Should have an ID after flush in service
    assert added_component.entity_id == sample_entity.id
    assert added_component.entity == sample_entity  # Relationship should be set

    retrieved_component = db_session.get(FileLocationComponent, added_component.id)
    assert retrieved_component is not None
    # Assert based on the fields used during creation
    assert retrieved_component.content_identifier == "test_content_id_123"  # Changed
    assert retrieved_component.physical_path_or_key == "physical/path/to/file.txt"  # Added
    assert retrieved_component.contextual_filename == "file.txt"  # Changed
    assert retrieved_component.storage_type == "test_local"


def test_add_component_to_non_existent_entity(db_session: Session):
    """Test adding a component to a non-existent entity raises ValueError."""
    non_existent_entity_id = 99999

    # Component instantiation would fail here if it strictly needs a valid entity_id/entity
    # for its own __init__ before the service is called.
    # This highlights the difficulty with the current component __init__ signatures
    # if the service is meant to handle *new, unassociated* components.
    # For this test, we can create a component that doesn't rely on entity_id for its own fields.
    # Let's use a simplified mock or a component that can be created "empty" for this.
    # However, FileLocationComponent needs entity_id for its __init__.

    # This test means the service should fail *before* trying to use component_instance too much.
    # The service calls get_entity(entity_id) first.

    # Create a dummy component instance that doesn't need entity_id for its own fields.
    # This is hard with kw_only=True on BaseComponent forcing entity_id.
    # We have to assume we can construct a component instance to pass.
    # The error should come from `get_entity`.

    # This component instance will be problematic to create if its __init__ requires entity_id.
    # Let's assume for the purpose of testing the service's entity check,
    # we can pass something that satisfies the type hint for component_instance.
    # This is a weakness in the test if component creation itself fails.

    # The service calls get_entity(entity_id) first. If that fails, component_instance isn't used much.
    # We use a mock that satisfies the BaseComponent type hint.
    mock_component = MagicMock(spec=BaseComponent)

    with pytest.raises(ValueError, match=f"Entity with ID {non_existent_entity_id} not found."):
        add_component_to_entity(db_session, non_existent_entity_id, mock_component)


def test_create_entity(db_session: Session):
    """Test creating a new entity."""
    entity = create_entity(db_session)
    db_session.commit()  # Commit to make it queryable through normal means if needed

    assert entity is not None
    assert entity.id is not None
    retrieved_entity = get_entity(db_session, entity.id)
    assert retrieved_entity is not None
    assert retrieved_entity.id == entity.id


def test_get_entity_non_existent(db_session: Session):
    """Test getting a non-existent entity returns None."""
    retrieved_entity = get_entity(db_session, 99999)
    assert retrieved_entity is None


def test_get_component_single(db_session: Session, sample_entity: Entity):
    """Test getting a single component that exists."""
    # Add a FilePropertiesComponent (which should be unique per entity in current setup)
    from dam.models import FilePropertiesComponent

    fpc_data = FilePropertiesComponent(
        entity_id=sample_entity.id,
        entity=sample_entity,
        original_filename="test.jpg",
        file_size_bytes=1024,
        mime_type="image/jpeg",
    )
    add_component_to_entity(db_session, sample_entity.id, fpc_data)
    db_session.commit()

    retrieved_fpc = get_component(db_session, sample_entity.id, FilePropertiesComponent)
    assert retrieved_fpc is not None
    assert retrieved_fpc.id == fpc_data.id
    assert retrieved_fpc.original_filename == "test.jpg"


def test_get_component_single_non_existent(db_session: Session, sample_entity: Entity):
    """Test getting a single component that does not exist for an entity."""
    from dam.models import (
        FilePropertiesComponent,
    )  # Ensure it's a different component type or not added

    retrieved_fpc = get_component(db_session, sample_entity.id, FilePropertiesComponent)
    assert retrieved_fpc is None


def test_get_components_multiple(db_session: Session, sample_entity: Entity):
    """Test getting multiple components of the same type for an entity."""
    loc1 = FileLocationComponent(
        entity_id=sample_entity.id,
        entity=sample_entity,
        content_identifier="content_id_loc1",  # Changed
        physical_path_or_key="path/loc1.txt",  # Added
        contextual_filename="loc1.txt",  # Changed
        storage_type="local",
    )
    loc2 = FileLocationComponent(
        entity_id=sample_entity.id,
        entity=sample_entity,
        content_identifier="content_id_loc2",  # Changed
        physical_path_or_key="path/loc2.txt",  # Added
        contextual_filename="loc2.txt",  # Changed
        storage_type="local",
    )
    add_component_to_entity(db_session, sample_entity.id, loc1)
    add_component_to_entity(db_session, sample_entity.id, loc2)
    db_session.commit()

    retrieved_locs = get_components(db_session, sample_entity.id, FileLocationComponent)
    assert len(retrieved_locs) == 2
    content_identifiers = {loc.content_identifier for loc in retrieved_locs}  # Changed
    contextual_filenames = {loc.contextual_filename for loc in retrieved_locs}  # Changed
    assert "content_id_loc1" in content_identifiers  # Changed
    assert "content_id_loc2" in content_identifiers  # Changed
    assert "loc1.txt" in contextual_filenames  # Changed
    assert "loc2.txt" in contextual_filenames  # Changed


def test_get_components_empty(db_session: Session, sample_entity: Entity):
    """Test getting components when none of that type exist for an entity."""
    retrieved_locs = get_components(db_session, sample_entity.id, FileLocationComponent)
    assert len(retrieved_locs) == 0


def test_remove_component(db_session: Session, sample_entity: Entity):
    """Test removing a component from an entity."""
    loc = FileLocationComponent(
        entity_id=sample_entity.id,
        entity=sample_entity,
        content_identifier="content_id_to_delete",  # Changed
        physical_path_or_key="path/to_delete.txt",  # Added
        contextual_filename="to_delete.txt",  # Changed
        storage_type="local",
    )
    add_component_to_entity(db_session, sample_entity.id, loc)
    db_session.commit()
    component_id_to_delete = loc.id

    # Verify it exists
    assert db_session.get(FileLocationComponent, component_id_to_delete) is not None

    remove_component(db_session, loc)
    db_session.commit()  # Commit deletion

    assert db_session.get(FileLocationComponent, component_id_to_delete) is None


def test_delete_entity_cascades_components(db_session: Session, sample_entity: Entity):
    """Test that deleting an entity also deletes its associated components."""
    from dam.models import (
        ContentHashSHA256Component,  # Changed from ContentHashComponent
    )  # Using another component type for variety

    # Add some components

    loc = FileLocationComponent(
        entity_id=sample_entity.id,
        entity=sample_entity,
        content_identifier="content_id_loc_del",  # Changed
        physical_path_or_key="path/loc_del.txt",  # Added
        contextual_filename="loc_del.txt",  # Changed
        storage_type="local",
    )
    chc_sha256 = ContentHashSHA256Component(  # Changed to a specific component
        entity_id=sample_entity.id,
        entity=sample_entity,
        hash_value="testhash_del_sha256",
    )

    add_component_to_entity(db_session, sample_entity.id, loc)
    add_component_to_entity(db_session, sample_entity.id, chc_sha256)  # Use the new variable
    db_session.commit()

    loc_id = loc.id
    chc_id = chc_sha256.id  # Use the new variable
    entity_id_to_delete = sample_entity.id

    # Verify components exist
    assert db_session.get(FileLocationComponent, loc_id) is not None
    assert db_session.get(ContentHashSHA256Component, chc_id) is not None  # Check specific component

    deleted = delete_entity(db_session, entity_id_to_delete)
    db_session.commit()

    assert deleted is True
    assert db_session.get(Entity, entity_id_to_delete) is None
    assert db_session.get(FileLocationComponent, loc_id) is None
    assert db_session.get(ContentHashSHA256Component, chc_id) is None  # Check specific component


def test_delete_non_existent_entity(db_session: Session):
    """Test deleting a non-existent entity returns False."""

    deleted = delete_entity(db_session, 99998)
    assert deleted is False


# Could add a test for IntegrityError if a component has a unique constraint
# (e.g., adding FilePropertiesComponent twice to the same entity if it has unique on entity_id)
# This would require FilePropertiesComponent to be tested here.
# The current add_component_to_entity does a flush, so it would catch this.

# Example: Test unique constraint on FilePropertiesComponent (if it's unique on entity_id)
# from dam.models import FilePropertiesComponent
# def test_add_component_violates_unique_constraint(db_session: Session, sample_entity: Entity):
#     prop1 = FilePropertiesComponent(entity_id=sample_entity.id, entity=sample_entity, original_filename="file1.txt")
#     add_component_to_entity(db_session, sample_entity.id, prop1)
#     db_session.commit() # First one is fine
#
#     prop2 = FilePropertiesComponent(entity_id=sample_entity.id, entity=sample_entity, original_filename="file2.txt")
#     with pytest.raises(IntegrityError):
#         add_component_to_entity(db_session, sample_entity.id, prop2)
#         # The flush is inside add_component_to_entity
#     db_session.rollback()

# The tests for `add_component_to_entity` reveal that if components (due to kw_only=True on Base)
# require entity_id and entity in their __init__, the service function is less about
# creating the association from scratch and more about adding a pre-associated or
# correctly initialized component to the session and flushing.
# The alternative is for the service to take component_type and component_data_dict.
# For now, the tests will work with the assumption that component_instance is
# already validly created (including its entity_id and entity relationship if required by its __init__).
# The service then re-affirms these by setting them from the passed entity_id.


# --- Tests for new query helpers ---


def test_find_entities_by_component_attribute_value(db_session: Session):
    # Setup: Create entities and components
    entity1 = create_entity(db_session)
    fpc1 = FilePropertiesComponent(
        entity_id=entity1.id, entity=entity1, original_filename="file1.jpg", mime_type="image/jpeg", file_size_bytes=100
    )
    add_component_to_entity(db_session, entity1.id, fpc1)

    entity2 = create_entity(db_session)
    fpc2 = FilePropertiesComponent(
        entity_id=entity2.id, entity=entity2, original_filename="file2.png", mime_type="image/png", file_size_bytes=200
    )
    add_component_to_entity(db_session, entity2.id, fpc2)

    entity3 = create_entity(db_session)
    fpc3 = FilePropertiesComponent(
        entity_id=entity3.id, entity=entity3, original_filename="file3.jpg", mime_type="image/jpeg", file_size_bytes=300
    )
    add_component_to_entity(db_session, entity3.id, fpc3)

    # Entity4 has no FilePropertiesComponent
    entity4 = create_entity(db_session)

    db_session.commit()

    # Test 1: Find entities with mime_type "image/jpeg"
    jpeg_entities = find_entities_by_component_attribute_value(
        db_session, FilePropertiesComponent, "mime_type", "image/jpeg"
    )
    assert len(jpeg_entities) == 2
    jpeg_entity_ids = {e.id for e in jpeg_entities}
    assert entity1.id in jpeg_entity_ids
    assert entity3.id in jpeg_entity_ids
    assert entity2.id not in jpeg_entity_ids
    assert entity4.id not in jpeg_entity_ids

    # Test 2: Find entities with mime_type "image/png"
    png_entities = find_entities_by_component_attribute_value(
        db_session, FilePropertiesComponent, "mime_type", "image/png"
    )
    assert len(png_entities) == 1
    assert png_entities[0].id == entity2.id

    # Test 3: Find entities with non-existent mime_type
    gif_entities = find_entities_by_component_attribute_value(
        db_session, FilePropertiesComponent, "mime_type", "image/gif"
    )
    assert len(gif_entities) == 0

    # Test 4: Find entities by file_size_bytes
    large_entities = find_entities_by_component_attribute_value(
        db_session, FilePropertiesComponent, "file_size_bytes", 300
    )
    assert len(large_entities) == 1
    assert large_entities[0].id == entity3.id

    # Test 5: Invalid attribute name
    with pytest.raises(AttributeError):
        find_entities_by_component_attribute_value(
            db_session, FilePropertiesComponent, "non_existent_attr", "image/jpeg"
        )

    # Test 6: Invalid component type
    with pytest.raises(TypeError):
        find_entities_by_component_attribute_value(
            db_session,
            Entity,
            "mime_type",
            "image/jpeg",  # type: ignore
        )


def test_find_entities_with_components(db_session: Session):
    # Setup
    entity1 = create_entity(db_session)  # Has FPC, IDC
    fpc1 = FilePropertiesComponent(
        entity_id=entity1.id, entity=entity1, original_filename="e1.jpg", mime_type="image/jpeg", file_size_bytes=100
    )
    idc1 = ImageDimensionsComponent(entity_id=entity1.id, entity=entity1, width=800, height=600)
    add_component_to_entity(db_session, entity1.id, fpc1)
    add_component_to_entity(db_session, entity1.id, idc1)

    entity2 = create_entity(db_session)  # Has FPC only
    fpc2 = FilePropertiesComponent(
        entity_id=entity2.id, entity=entity2, original_filename="e2.txt", mime_type="text/plain", file_size_bytes=50
    )
    add_component_to_entity(db_session, entity2.id, fpc2)

    entity3 = create_entity(db_session)  # Has IDC only
    idc3 = ImageDimensionsComponent(entity_id=entity3.id, entity=entity3, width=1024, height=768)
    add_component_to_entity(db_session, entity3.id, idc3)

    entity4 = create_entity(db_session)  # Has FPC, IDC (different values)
    fpc4 = FilePropertiesComponent(
        entity_id=entity4.id, entity=entity4, original_filename="e4.png", mime_type="image/png", file_size_bytes=150
    )
    idc4 = ImageDimensionsComponent(entity_id=entity4.id, entity=entity4, width=300, height=200)
    add_component_to_entity(db_session, entity4.id, fpc4)
    add_component_to_entity(db_session, entity4.id, idc4)

    entity5 = create_entity(db_session)  # No relevant components

    db_session.commit()

    # Test 1: Find entities with FPC AND IDC
    entities_with_both = find_entities_with_components(db_session, [FilePropertiesComponent, ImageDimensionsComponent])
    assert len(entities_with_both) == 2
    both_ids = {e.id for e in entities_with_both}
    assert entity1.id in both_ids
    assert entity4.id in both_ids
    assert entity2.id not in both_ids
    assert entity3.id not in both_ids
    assert entity5.id not in both_ids

    # Test 2: Find entities with FPC only
    entities_with_fpc = find_entities_with_components(db_session, [FilePropertiesComponent])
    assert len(entities_with_fpc) == 3  # e1, e2, e4
    fpc_ids = {e.id for e in entities_with_fpc}
    assert entity1.id in fpc_ids
    assert entity2.id in fpc_ids
    assert entity4.id in fpc_ids

    # Test 3: Find entities with IDC only
    entities_with_idc = find_entities_with_components(db_session, [ImageDimensionsComponent])
    assert len(entities_with_idc) == 3  # e1, e3, e4
    idc_ids = {e.id for e in entities_with_idc}
    assert entity1.id in idc_ids
    assert entity3.id in idc_ids
    assert entity4.id in idc_ids

    # Test 4: Empty list of required components
    assert find_entities_with_components(db_session, []) == []

    # Test 5: Non-existent component type (hypothetical)
    class NonExistentComponent(BaseComponent):  # type: ignore
        __tablename__ = "non_existent_component_test"
        # Minimal fields for BaseComponent if needed by test setup
        # However, this component won't have a table.
        # The join would fail. SQLAlchemy might raise error before query, or DB will.
        # Let's make it a valid but unadded component type.

    # This test is more about how SQLAlchemy handles joins to tables that might exist but have no matching entries.
    # If NonExistentComponent is not a real table, this will error out earlier.
    # Let's use a real component that simply isn't added to any entities.

    entities_with_audio = find_entities_with_components(db_session, [AudioPropertiesComponent])
    assert len(entities_with_audio) == 0

    entities_with_fpc_and_audio = find_entities_with_components(
        db_session, [FilePropertiesComponent, AudioPropertiesComponent]
    )
    assert len(entities_with_fpc_and_audio) == 0

    # Test 6: Invalid component type in list
    with pytest.raises(TypeError):
        find_entities_with_components(db_session, [FilePropertiesComponent, Entity])  # type: ignore
