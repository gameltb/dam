import pytest  # noqa F401
from sqlalchemy.orm import Session
from dam.models import ImageDimensionsComponent
from dam.services.ecs_service import create_entity, add_component_to_entity, get_component


def test_create_image_dimensions_component(db_session: Session):
    """Test creating an ImageDimensionsComponent and associating it with an entity."""
    entity = create_entity(db_session)
    db_session.commit()

    dimensions_data = {
        "width_pixels": 1920,
        "height_pixels": 1080,
    }
    dimensions_component = ImageDimensionsComponent(entity_id=entity.id, entity=entity, **dimensions_data)
    added_component = add_component_to_entity(db_session, entity.id, dimensions_component)
    db_session.commit()

    assert added_component.id is not None
    assert added_component.entity_id == entity.id
    assert added_component.width_pixels == 1920
    assert added_component.height_pixels == 1080

    retrieved_component = get_component(db_session, entity.id, ImageDimensionsComponent)
    assert retrieved_component is not None
    assert retrieved_component.id == added_component.id
    assert retrieved_component.width_pixels == 1920

    assert repr(added_component).startswith("<ImageDimensionsComponent")


def test_image_dimensions_component_nullable_fields(db_session: Session):
    """Test creating an ImageDimensionsComponent with nullable fields explicitly set to None."""
    entity = create_entity(db_session)
    db_session.commit()

    # These fields have default=None in the model, so not passing them means they are None
    dimensions_component = ImageDimensionsComponent(entity_id=entity.id, entity=entity)
    added_component = add_component_to_entity(db_session, entity.id, dimensions_component)
    db_session.commit()

    assert added_component.width_pixels is None
    assert added_component.height_pixels is None

    retrieved_component = get_component(db_session, entity.id, ImageDimensionsComponent)
    assert retrieved_component is not None
    assert retrieved_component.width_pixels is None
    assert retrieved_component.height_pixels is None


def test_image_dimensions_component_partial_data(db_session: Session):
    """Test creating an ImageDimensionsComponent with some fields None."""
    entity = create_entity(db_session)
    db_session.commit()

    dimensions_data = {
        "width_pixels": 720,
        "height_pixels": None,  # Explicitly None
    }
    dimensions_component = ImageDimensionsComponent(entity_id=entity.id, entity=entity, **dimensions_data)
    added_component = add_component_to_entity(db_session, entity.id, dimensions_component)
    db_session.commit()

    assert added_component.width_pixels == 720
    assert added_component.height_pixels is None

    retrieved_component = get_component(db_session, entity.id, ImageDimensionsComponent)
    assert retrieved_component is not None
    assert retrieved_component.width_pixels == 720
    assert retrieved_component.height_pixels is None
