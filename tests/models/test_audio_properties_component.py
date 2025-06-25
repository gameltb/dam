import pytest  # noqa F401
from sqlalchemy.orm import Session
from dam.models import AudioPropertiesComponent
from dam.services.ecs_service import create_entity, add_component_to_entity, get_component


def test_create_audio_properties_component(db_session: Session):
    """Test creating an AudioPropertiesComponent and associating it with an entity."""
    entity = create_entity(db_session)
    db_session.commit()

    audio_props_data = {
        "duration_seconds": 300.25,
        "codec_name": "mp3",
        "sample_rate_hz": 44100,
        "channels": 2,
        "bit_rate_kbps": 192,
    }
    # Pass the entity object itself for the relationship
    audio_component = AudioPropertiesComponent(entity_id=entity.id, entity=entity, **audio_props_data)
    added_component = add_component_to_entity(db_session, entity.id, audio_component)
    db_session.commit()

    assert added_component.id is not None
    assert added_component.entity_id == entity.id
    for key, value in audio_props_data.items():
        assert getattr(added_component, key) == value

    retrieved_component = get_component(db_session, entity.id, AudioPropertiesComponent)
    assert retrieved_component is not None
    assert retrieved_component.id == added_component.id
    assert retrieved_component.codec_name == "mp3"

    assert repr(added_component).startswith("<AudioPropertiesComponent")


def test_audio_properties_component_nullable_fields(db_session: Session):
    """Test creating an AudioPropertiesComponent with nullable fields set to None."""
    entity = create_entity(db_session)
    db_session.commit()

    audio_props_data = {
        "duration_seconds": None,
        "codec_name": "flac",
        "sample_rate_hz": 48000,
        "channels": None,
        "bit_rate_kbps": None,
    }
    audio_component = AudioPropertiesComponent(entity_id=entity.id, entity=entity, **audio_props_data)
    added_component = add_component_to_entity(db_session, entity.id, audio_component)
    db_session.commit()

    assert added_component.duration_seconds is None
    assert added_component.channels is None
    assert added_component.bit_rate_kbps is None
    assert added_component.codec_name == "flac"

    retrieved_component = get_component(db_session, entity.id, AudioPropertiesComponent)
    assert retrieved_component is not None
    assert retrieved_component.channels is None
    assert retrieved_component.bit_rate_kbps is None
