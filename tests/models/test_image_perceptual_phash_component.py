import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dam.models.entity import Entity
from dam.models.image_perceptual_hash_phash_component import ImagePerceptualPHashComponent


# Re-using the test_entity fixture from conftest or defining it locally if preferred.
# For simplicity here, assuming test_entity can be imported or is available via conftest.
# If not, it would be:
@pytest.fixture
def test_entity(db_session: Session) -> Entity:
    """Fixture to create and commit an entity for use in component tests."""
    entity = Entity()
    db_session.add(entity)
    db_session.commit()
    return entity


def test_create_image_perceptual_phash_component_instance(test_entity: Entity):
    """Test basic instantiation of an ImagePerceptualPHashComponent."""
    iphc = ImagePerceptualPHashComponent(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="perceptual_hash_string_abc123",
    )
    assert iphc.entity_id == test_entity.id
    assert iphc.hash_value == "perceptual_hash_string_abc123"


def test_add_and_retrieve_image_perceptual_phash_component(db_session: Session, test_entity: Entity):
    """Test adding and retrieving an ImagePerceptualPHashComponent."""
    iphc = ImagePerceptualPHashComponent(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="phash_val_456",
    )
    db_session.add(iphc)
    db_session.commit()

    assert iphc.id is not None
    retrieved_iphc = db_session.get(ImagePerceptualPHashComponent, iphc.id)
    assert retrieved_iphc is not None
    assert retrieved_iphc.entity_id == test_entity.id
    assert retrieved_iphc.hash_value == "phash_val_456"


def test_image_perceptual_phash_component_relationship_to_entity(db_session: Session, test_entity: Entity):
    """Test the relationship from the component back to the entity."""
    iphc = ImagePerceptualPHashComponent(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="ahash_for_relation_test",  # This test will be specific to pHash, value can be generic
    )
    db_session.add(iphc)
    db_session.commit()
    db_session.refresh(iphc)

    assert iphc.entity is not None
    assert iphc.entity.id == test_entity.id
    assert iphc.entity == test_entity


def test_image_perceptual_phash_component_unique_constraint(db_session: Session, test_entity: Entity):
    """Test the unique constraint (entity_id, hash_value)."""
    iphc1 = ImagePerceptualPHashComponent(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="first_phash_value",
    )
    db_session.add(iphc1)
    db_session.commit()

    iphc2 = ImagePerceptualPHashComponent(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="first_phash_value",  # Same entity_id and hash_value
    )
    db_session.add(iphc2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Verify that a different hash_value for the same entity is allowed
    iphc3 = ImagePerceptualPHashComponent(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="another_phash_value",
    )
    db_session.add(iphc3)
    db_session.commit()
    assert iphc3.id is not None

    # Verify that same hash_value for a different entity is allowed
    another_entity = Entity()
    db_session.add(another_entity)
    db_session.commit()

    iphc4 = ImagePerceptualPHashComponent(
        entity_id=another_entity.id,  # type: ignore
        entity=another_entity,  # Added entity object
        hash_value="first_phash_value",  # Same hash_value as chc1 but different entity
    )
    db_session.add(iphc4)
    db_session.commit()
    assert iphc4.id is not None


def test_delete_image_perceptual_phash_component(db_session: Session, test_entity: Entity):
    """Test deleting an ImagePerceptualPHashComponent."""
    iphc = ImagePerceptualPHashComponent(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,  # Added entity object
        hash_value="phash_to_delete",
    )
    db_session.add(iphc)
    db_session.commit()

    component_id = iphc.id
    assert db_session.get(ImagePerceptualPHashComponent, component_id) is not None

    db_session.delete(iphc)
    db_session.commit()
    assert db_session.get(ImagePerceptualPHashComponent, component_id) is None
