import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dam.models.content_hash_md5_component import ContentHashMD5Component
from dam.models.entity import Entity


@pytest.fixture
def test_entity(db_session: Session) -> Entity:
    """Fixture to create and commit an entity for use in component tests."""
    entity = Entity()
    db_session.add(entity)
    db_session.commit()
    return entity


def test_create_content_hash_md5_component_instance(test_entity: Entity):
    """Test basic instantiation of a ContentHashMD5Component."""
    chc = ContentHashMD5Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,
        hash_value="a_hash_string_representing_md5",
    )
    assert chc.entity_id == test_entity.id
    assert chc.hash_value == "a_hash_string_representing_md5"


def test_add_and_retrieve_content_hash_md5_component(db_session: Session, test_entity: Entity):
    """Test adding and retrieving a ContentHashMD5Component."""
    chc = ContentHashMD5Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,
        hash_value="md5hash123",
    )
    db_session.add(chc)
    db_session.commit()

    assert chc.id is not None
    retrieved_chc = db_session.get(ContentHashMD5Component, chc.id)
    assert retrieved_chc is not None
    assert retrieved_chc.entity_id == test_entity.id
    assert retrieved_chc.hash_value == "md5hash123"


def test_content_hash_md5_component_relationship_to_entity(db_session: Session, test_entity: Entity):
    """Test the relationship from the component back to the entity."""
    chc = ContentHashMD5Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,
        hash_value="md5_hash_for_relation_test",
    )
    db_session.add(chc)
    db_session.commit()
    db_session.refresh(chc)

    assert chc.entity is not None
    assert chc.entity.id == test_entity.id
    assert chc.entity == test_entity


def test_content_hash_md5_component_unique_constraint(db_session: Session, test_entity: Entity):
    """Test the unique constraint (entity_id, hash_value)."""
    chc1 = ContentHashMD5Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,
        hash_value="first_md5_hash_value",
    )
    db_session.add(chc1)
    db_session.commit()

    chc2 = ContentHashMD5Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,
        hash_value="first_md5_hash_value",
    )
    db_session.add(chc2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    chc3 = ContentHashMD5Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,
        hash_value="another_md5_hash_value",
    )
    db_session.add(chc3)
    db_session.commit()
    assert chc3.id is not None

    another_entity = Entity()
    db_session.add(another_entity)
    db_session.commit()

    chc4 = ContentHashMD5Component(
        entity_id=another_entity.id,  # type: ignore
        entity=another_entity,
        hash_value="first_md5_hash_value",
    )
    db_session.add(chc4)
    db_session.commit()
    assert chc4.id is not None


def test_delete_content_hash_md5_component(db_session: Session, test_entity: Entity):
    """Test deleting a ContentHashMD5Component."""
    chc = ContentHashMD5Component(
        entity_id=test_entity.id,  # type: ignore
        entity=test_entity,
        hash_value="md5_hash_to_delete",
    )
    db_session.add(chc)
    db_session.commit()

    component_id = chc.id
    assert db_session.get(ContentHashMD5Component, component_id) is not None

    db_session.delete(chc)
    db_session.commit()
    assert db_session.get(ContentHashMD5Component, component_id) is None
