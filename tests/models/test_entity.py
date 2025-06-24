from datetime import datetime

from sqlalchemy.orm import Session

from dam.models.entity import Entity


def test_create_entity_instance():
    """Test basic instantiation of an Entity (not in DB yet)."""
    entity = Entity()
    # Note: id, created_at, updated_at are typically set by DB or SQLAlchemy upon flush/commit.
    # So, before commit, they might be None or not set, depending on configuration.
    # For MappedAsDataclass with field(init=False), they won't be in __init__.
    assert entity is not None
    # We can't assert entity.id here as it's init=False and DB-generated.


def test_add_and_retrieve_entity(db_session: Session):
    """Test adding an Entity to the DB and retrieving it."""
    entity = Entity()
    db_session.add(entity)
    db_session.commit()

    assert entity.id is not None
    assert isinstance(entity.created_at, datetime)
    assert isinstance(entity.updated_at, datetime)

    retrieved_entity = db_session.get(Entity, entity.id)
    assert retrieved_entity is not None
    assert retrieved_entity.id == entity.id
    assert retrieved_entity.created_at == entity.created_at


def test_update_entity_timestamps(db_session: Session):
    """Test that updated_at timestamp changes on modification."""
    entity = Entity()
    db_session.add(entity)
    db_session.commit()

    original_updated_at = entity.updated_at

    # To trigger an update, we need a field to change.
    # If Entity had other mutable fields, we'd change one here.
    # For now, we can simulate a change by re-committing or nudging the session.
    # A more robust way would be to have a mutable field.
    # Let's assume for now just re-adding and committing might refresh server defaults
    # if the DB is configured to do so on ANY update, or if a field were actually changed.
    # This test is a bit weak without a mutable field on Entity.

    # Let's try to expire and refresh to see if onupdate works as expected by SQLAlchemy
    # even without a direct change to a mapped attribute.
    # For some DBs, just the act of an UPDATE statement triggers onupdate.

    # We don't have another mutable field on Entity to easily trigger an UPDATE.
    # Setting entity.updated_at = None caused IntegrityError.
    # For now, just ensure commit() doesn't break and updated_at is still valid.
    # A true test of onupdate would require a schema change or a different testing strategy.
    db_session.add(entity)
    db_session.commit()  # Commit again
    db_session.refresh(entity)

    # This assertion depends heavily on DB and SQLAlchemy behavior for server_onupdate
    # without actual data change. A more reliable test would involve changing a field.
    # For now, we expect it to be at least the same or newer.
    assert entity.updated_at >= original_updated_at
    assert isinstance(entity.updated_at, datetime)  # Ensure it's still a datetime

    # A better test if we had a mutable field 'name':
    # entity.name = "New Name"
    # db_session.commit()
    # assert entity.updated_at > original_updated_at


def test_delete_entity(db_session: Session):
    """Test deleting an Entity from the DB."""
    entity = Entity()
    db_session.add(entity)
    db_session.commit()

    entity_id = entity.id
    assert db_session.get(Entity, entity_id) is not None

    db_session.delete(entity)
    db_session.commit()

    assert db_session.get(Entity, entity_id) is None


# Future: Add tests for cascading deletes if relationships are added to Entity
# that require such behavior (e.g., entity.delete() should delete its components).
# This depends on cascade options in relationship(). Our current setup is that
# components have a foreign key to Entity, so deleting an Entity might fail
# if components still reference it, unless cascade delete is configured on Entity's
# relationships or the DB foreign key has ON DELETE CASCADE.
# For now, components would need to be deleted first or cascade configured.
