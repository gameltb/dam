import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dam.models.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.entity import Entity


@pytest.fixture
def test_entity(db_session: Session) -> Entity:
    """Fixture to create and commit an entity for use in component tests."""
    entity = Entity()
    db_session.add(entity)
    db_session.commit()
    return entity


def test_create_content_hash_sha256_component_instance(test_entity: Entity):
    """Test basic instantiation of a ContentHashSHA256Component."""
    chc = ContentHashSHA256Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="a_very_long_hash_string_representing_sha256",
    )
    assert chc.entity_id == test_entity.id
    assert chc.hash_value == "a_very_long_hash_string_representing_sha256"


def test_add_and_retrieve_content_hash_sha256_component(db_session: Session, test_entity: Entity):
    """Test adding and retrieving a ContentHashSHA256Component."""
    chc = ContentHashSHA256Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="hash123",
    )
    db_session.add(chc)
    db_session.commit()

    assert chc.id is not None
    retrieved_chc = db_session.get(ContentHashSHA256Component, chc.id)
    assert retrieved_chc is not None
    assert retrieved_chc.entity_id == test_entity.id
    assert retrieved_chc.hash_value == "hash123"


def test_content_hash_sha256_component_relationship_to_entity(db_session: Session, test_entity: Entity):
    """Test the relationship from the component back to the entity."""
    chc = ContentHashSHA256Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="hash_for_relation_test",
    )
    db_session.add(chc)
    db_session.commit()
    db_session.refresh(chc)  # Ensure relationship is loaded

    assert chc.entity is not None
    assert chc.entity.id == test_entity.id
    assert chc.entity == test_entity  # Should be the same object if session is consistent


def test_content_hash_sha256_component_unique_constraint(db_session: Session, test_entity: Entity):
    """Test the unique constraint (entity_id, hash_value)."""
    chc1 = ContentHashSHA256Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="first_hash_value",
    )
    db_session.add(chc1)
    db_session.commit()

    chc2 = ContentHashSHA256Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="first_hash_value",  # Same entity_id and hash_value
    )
    db_session.add(chc2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()  # Important to rollback after expected error

    # Verify that a different hash_value for the same entity is allowed
    chc3 = ContentHashSHA256Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="another_sha256_hash_value",
    )
    db_session.add(chc3)
    # This should fail because test_entity.id is already associated with chc1
    with pytest.raises(IntegrityError, match="UNIQUE constraint failed: component_content_hash_sha256.entity_id"):
        db_session.commit()
    db_session.rollback()

    # Verify that same hash_value for a different entity also fails due to uq_sha256_hash_value
    another_entity = Entity()
    db_session.add(another_entity)
    db_session.commit()

    chc4 = ContentHashSHA256Component(
        entity_id=another_entity.id,  # type: ignore
        entity=another_entity,  # Added entity object
        hash_value="first_hash_value",  # Same hash_value as chc1, should violate uq_sha256_hash_value
    )
    db_session.add(chc4)
    with pytest.raises(IntegrityError, match="UNIQUE constraint failed: component_content_hash_sha256.hash_value"):
        db_session.commit()
    db_session.rollback()

    # Verify that a new entity with a new hash is fine
    yet_another_entity = Entity()
    db_session.add(yet_another_entity)
    db_session.commit()
    chc5 = ContentHashSHA256Component(
        entity_id=yet_another_entity.id,  # type: ignore
        entity=yet_another_entity,
        hash_value="completely_new_hash_value",
    )
    db_session.add(chc5)
    db_session.commit()
    assert chc5.id is not None


def test_delete_content_hash_sha256_component(db_session: Session, test_entity: Entity):
    """Test deleting a ContentHashSHA256Component."""
    chc = ContentHashSHA256Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="hash_to_delete",
    )
    db_session.add(chc)
    db_session.commit()

    component_id = chc.id
    assert db_session.get(ContentHashSHA256Component, component_id) is not None

    db_session.delete(chc)
    db_session.commit()
    assert db_session.get(ContentHashSHA256Component, component_id) is None


# Note on type: ignore for entity_id:
# BaseComponent.entity_id is Mapped[int]. When creating a component instance,
# we pass test_entity.id which is also an int (or could be None before commit, but
# test_entity fixture ensures it's committed and has an ID).
# The type checker might be overly cautious or there might be a subtle typing nuance.
# For practical purposes in these tests, test_entity.id is a valid integer ID.
# If entity_id was `Mapped["Entity"]` then we'd pass `entity=test_entity`.
# Since it's `Mapped[int]` and `ForeignKey("entities.id")`, passing the ID is correct.
# The `type: ignore` is a pragmatic choice here if the type checker complains,
# assuming the underlying logic is sound.
# A cleaner way might be to define `entity: Mapped["Entity"]` in BaseComponent and use that
# in the constructor, with `entity_id` being `mapped_column(ForeignKey(Entity.id))`.
# However, the current `BaseComponent` is designed with `entity_id: Mapped[int]`.
