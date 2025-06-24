import pytest
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError # For testing constraint violations
from unittest.mock import MagicMock # Moved import to top

from dam.models import Entity, FileLocationComponent # Using FileLocation as a sample component
from dam.services.ecs_service import add_component_to_entity, get_entity
from dam.models.base_component import BaseComponent # For type hinting

# Fixture to create a sample entity, relies on db_session from conftest.py
@pytest.fixture
def sample_entity(db_session: Session) -> Entity:
    entity = Entity()
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity) # Ensure it has its ID etc.
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
        filepath="/path/to/file.txt",
        storage_type="test_local"
    )

    # The service will re-set entity_id and entity, which is fine.
    added_component = add_component_to_entity(db_session, sample_entity.id, component_to_add)
    db_session.commit() # Commit after service call

    assert added_component is not None
    assert added_component.id is not None # Should have an ID after flush in service
    assert added_component.entity_id == sample_entity.id
    assert added_component.entity == sample_entity # Relationship should be set

    retrieved_component = db_session.get(FileLocationComponent, added_component.id)
    assert retrieved_component is not None
    assert retrieved_component.filepath == "/path/to/file.txt"
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

    from unittest.mock import MagicMock # Ensure this import is at the top of the file or accessible

    # The service calls get_entity(entity_id) first. If that fails, component_instance isn't used much.
    # We use a mock that satisfies the BaseComponent type hint.
    mock_component = MagicMock(spec=BaseComponent)

    with pytest.raises(ValueError, match=f"Entity with ID {non_existent_entity_id} not found."):
        add_component_to_entity(db_session, non_existent_entity_id, mock_component)


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
