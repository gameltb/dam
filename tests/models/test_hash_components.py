import pytest
import binascii
from sqlalchemy.orm import Session

from dam.models import Entity
from dam.models.content_hash_md5_component import ContentHashMD5Component
from dam.models.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.image_perceptual_hash_phash_component import ImagePerceptualPHashComponent
from dam.models.image_perceptual_hash_ahash_component import ImagePerceptualAHashComponent
from dam.models.image_perceptual_hash_dhash_component import ImagePerceptualDHashComponent
from dam.services import ecs_service


@pytest.fixture
def asset_entity(db_session: Session) -> Entity:
    entity = ecs_service.create_entity(db_session)
    db_session.commit()
    return entity

def test_content_hash_md5_bytes_storage(db_session: Session, asset_entity: Entity):
    """Test ContentHashMD5Component stores and retrieves bytes."""
    md5_hex = "098f6bcd4621d373cade4e832627b4f6" # md5 for "test"
    md5_bytes = binascii.unhexlify(md5_hex)

    md5_comp = ContentHashMD5Component(entity_id=asset_entity.id, hash_value=md5_bytes)
    ecs_service.add_component_to_entity(db_session, asset_entity.id, md5_comp)
    db_session.commit()

    retrieved_comp = db_session.get(ContentHashMD5Component, md5_comp.id)
    assert retrieved_comp is not None
    assert isinstance(retrieved_comp.hash_value, bytes)
    assert retrieved_comp.hash_value == md5_bytes
    assert retrieved_comp.hash_value.hex() == md5_hex

def test_content_hash_sha256_bytes_storage(db_session: Session, asset_entity: Entity):
    """Test ContentHashSHA256Component stores and retrieves bytes."""
    sha256_hex = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08" # sha256 for "test"
    sha256_bytes = binascii.unhexlify(sha256_hex)

    sha256_comp = ContentHashSHA256Component(entity_id=asset_entity.id, hash_value=sha256_bytes)
    ecs_service.add_component_to_entity(db_session, asset_entity.id, sha256_comp)
    db_session.commit()

    retrieved_comp = db_session.get(ContentHashSHA256Component, sha256_comp.id)
    assert retrieved_comp is not None
    assert isinstance(retrieved_comp.hash_value, bytes)
    assert retrieved_comp.hash_value == sha256_bytes
    assert retrieved_comp.hash_value.hex() == sha256_hex

def test_perceptual_phash_bytes_storage(db_session: Session, asset_entity: Entity):
    """Test ImagePerceptualPHashComponent stores and retrieves bytes."""
    phash_hex = "f0f0f0f0f0f0f0f0" # Example 64-bit pHash
    phash_bytes = binascii.unhexlify(phash_hex)

    phash_comp = ImagePerceptualPHashComponent(entity_id=asset_entity.id, hash_value=phash_bytes)
    ecs_service.add_component_to_entity(db_session, asset_entity.id, phash_comp)
    db_session.commit()

    retrieved_comp = db_session.get(ImagePerceptualPHashComponent, phash_comp.id)
    assert retrieved_comp is not None
    assert isinstance(retrieved_comp.hash_value, bytes)
    assert retrieved_comp.hash_value == phash_bytes
    assert retrieved_comp.hash_value.hex() == phash_hex
    assert len(retrieved_comp.hash_value) == 8 # 64 bits = 8 bytes

def test_perceptual_ahash_bytes_storage(db_session: Session, asset_entity: Entity):
    """Test ImagePerceptualAHashComponent stores and retrieves bytes."""
    ahash_hex = "0123456789abcdef" # Example 64-bit aHash
    ahash_bytes = binascii.unhexlify(ahash_hex)

    ahash_comp = ImagePerceptualAHashComponent(entity_id=asset_entity.id, hash_value=ahash_bytes)
    ecs_service.add_component_to_entity(db_session, asset_entity.id, ahash_comp)
    db_session.commit()

    retrieved_comp = db_session.get(ImagePerceptualAHashComponent, ahash_comp.id)
    assert retrieved_comp is not None
    assert isinstance(retrieved_comp.hash_value, bytes)
    assert retrieved_comp.hash_value == ahash_bytes
    assert retrieved_comp.hash_value.hex() == ahash_hex
    assert len(retrieved_comp.hash_value) == 8

def test_perceptual_dhash_bytes_storage(db_session: Session, asset_entity: Entity):
    """Test ImagePerceptualDHashComponent stores and retrieves bytes."""
    dhash_hex = "fedcba9876543210" # Example 64-bit dHash
    dhash_bytes = binascii.unhexlify(dhash_hex)

    dhash_comp = ImagePerceptualDHashComponent(entity_id=asset_entity.id, hash_value=dhash_bytes)
    ecs_service.add_component_to_entity(db_session, asset_entity.id, dhash_comp)
    db_session.commit()

    retrieved_comp = db_session.get(ImagePerceptualDHashComponent, dhash_comp.id)
    assert retrieved_comp is not None
    assert isinstance(retrieved_comp.hash_value, bytes)
    assert retrieved_comp.hash_value == dhash_bytes
    assert retrieved_comp.hash_value.hex() == dhash_hex
    assert len(retrieved_comp.hash_value) == 8
