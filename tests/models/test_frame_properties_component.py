import pytest  # noqa F401
from sqlalchemy.orm import Session
from dam.models import FramePropertiesComponent
from dam.services.ecs_service import create_entity, add_component_to_entity, get_component


def test_create_frame_properties_component(db_session: Session):
    """Test creating a FramePropertiesComponent and associating it with an entity."""
    entity = create_entity(db_session)
    db_session.commit()

    frame_props_data = {
        "frame_count": 150,
        "nominal_frame_rate": 10.0,
        "animation_duration_seconds": 15.0,
    }
    frame_component = FramePropertiesComponent(entity_id=entity.id, entity=entity, **frame_props_data)
    added_component = add_component_to_entity(db_session, entity.id, frame_component)
    db_session.commit()

    assert added_component.id is not None
    assert added_component.entity_id == entity.id
    for key, value in frame_props_data.items():
        assert getattr(added_component, key) == value

    retrieved_component = get_component(db_session, entity.id, FramePropertiesComponent)
    assert retrieved_component is not None
    assert retrieved_component.id == added_component.id
    assert retrieved_component.frame_count == 150

    assert repr(added_component).startswith("<FramePropertiesComponent")


def test_frame_properties_component_nullable_fields(db_session: Session):
    """Test creating a FramePropertiesComponent with nullable fields set to None."""
    entity = create_entity(db_session)
    db_session.commit()

    frame_props_data = {
        "frame_count": None,
        "nominal_frame_rate": None,
        "animation_duration_seconds": None,
    }
    frame_component = FramePropertiesComponent(entity_id=entity.id, entity=entity, **frame_props_data)
    added_component = add_component_to_entity(db_session, entity.id, frame_component)
    db_session.commit()

    assert added_component.frame_count is None
    assert added_component.nominal_frame_rate is None
    assert added_component.animation_duration_seconds is None

    retrieved_component = get_component(db_session, entity.id, FramePropertiesComponent)
    assert retrieved_component is not None
    assert retrieved_component.frame_count is None
    assert retrieved_component.nominal_frame_rate is None
