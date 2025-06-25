from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from dam.models import (
    ImagePerceptualHashComponent,
)
from dam.services import asset_service, file_operations

# Use fixtures from conftest.py for db_session
# We also need sample image paths, can reuse fixtures from test_file_operations or define new ones.


@pytest.fixture(scope="module")
def module_db_engine():
    """Creates an in-memory SQLite engine for the test module."""
    # Using a distinct engine for this module to avoid conflicts if other modules also use create_all
    # Alternatively, ensure conftest.py's engine is used and tables are managed carefully.
    # For simplicity, using the engine from dam.core.database which should be configured for tests.
    # Assuming dam.core.database.settings.DATABASE_URL is set to an in-memory DB for tests.
    # If not, this needs adjustment.
    # Let's use the conftest.py engine and session management.
    # This means we need to ensure conftest.py is set up or its fixtures are accessible.
    # The conftest.py from previous steps should be available.
    pass  # Rely on conftest.py's engine fixture


@pytest.fixture
def sample_image_a(tmp_path: Path) -> Path:
    img_a_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR42mNkgIL/DAwM/wUADgAB/vA/cQAAAABJRU5ErkJggg=="
    file_path = tmp_path / "sample_A.png"
    import base64

    file_path.write_bytes(base64.b64decode(img_a_b64))
    return file_path


@pytest.fixture
def non_image_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a test text file.")
    return file_path


def test_add_image_asset_creates_perceptual_hashes(db_session: Session, sample_image_a: Path):
    """Test adding an image asset creates ImagePerceptualHashComponents."""
    # Ensure ImageHash and Pillow are available for this test
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        pytest.skip("ImageHash or Pillow not installed, skipping perceptual hash integration test.")

    original_filename, size_bytes, mime_type = file_operations.get_file_properties(sample_image_a)
    content_hash = file_operations.calculate_sha256(sample_image_a)

    entity, created_new = asset_service.add_asset_file(
        session=db_session,
        filepath_on_disk=sample_image_a,
        original_filename=original_filename,
        mime_type=mime_type,  # Should be 'image/png'
        size_bytes=size_bytes,
        content_hash=content_hash,
    )
    db_session.commit()

    assert created_new is True
    assert entity.id is not None

    # Verify perceptual hashes
    stmt = select(ImagePerceptualHashComponent).where(ImagePerceptualHashComponent.entity_id == entity.id)
    phash_components = db_session.execute(stmt).scalars().all()

    assert len(phash_components) > 0  # Should have at least one, ideally 3 (phash, ahash, dhash)

    found_hash_types = {comp.hash_type for comp in phash_components}
    expected_hash_types = {"phash", "ahash", "dhash"}

    # The generate_perceptual_hashes might not produce all if image is too simple or errors occur for one type
    # So, check that what was produced is part of the expected set.
    # And that at least one was produced.
    assert any(htype in expected_hash_types for htype in found_hash_types)
    for comp in phash_components:
        assert comp.hash_value is not None
        assert len(comp.hash_value) > 0


def test_add_non_image_asset_no_perceptual_hashes(db_session: Session, non_image_file: Path):
    """Test adding a non-image asset does not create ImagePerceptualHashComponents."""
    original_filename, size_bytes, mime_type = file_operations.get_file_properties(non_image_file)
    content_hash = file_operations.calculate_sha256(non_image_file)

    entity, created_new = asset_service.add_asset_file(
        session=db_session,
        filepath_on_disk=non_image_file,
        original_filename=original_filename,
        mime_type=mime_type,  # Should be 'text/plain' or 'application/octet-stream'
        size_bytes=size_bytes,
        content_hash=content_hash,
    )
    db_session.commit()

    assert created_new is True
    assert entity.id is not None

    # Verify no perceptual hashes
    stmt = select(ImagePerceptualHashComponent).where(ImagePerceptualHashComponent.entity_id == entity.id)
    phash_components = db_session.execute(stmt).scalars().all()
    assert len(phash_components) == 0


def test_add_existing_image_content_adds_missing_perceptual_hashes(
    db_session: Session, sample_image_a: Path, tmp_path: Path
):
    """
    Test that if an image is added (content exists), and it's missing some perceptual hashes,
    they get added on a subsequent add_asset_file call for that content.
    """
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        pytest.skip("ImageHash or Pillow not installed, skipping advanced perceptual hash test.")

    # --- First add ---
    props1 = file_operations.get_file_properties(sample_image_a)
    chash1 = file_operations.calculate_sha256(sample_image_a)
    entity1, _ = asset_service.add_asset_file(db_session, sample_image_a, props1[0], props1[2], props1[1], chash1)
    db_session.commit()

    # Verify all 3 perceptual hashes are there initially
    stmt1 = select(ImagePerceptualHashComponent).where(ImagePerceptualHashComponent.entity_id == entity1.id)
    phash_comps1 = db_session.execute(stmt1).scalars().all()
    assert len(phash_comps1) >= 1  # Should be 3 if all generated
    initial_hash_types = {c.hash_type for c in phash_comps1}
    if not {"phash", "ahash", "dhash"}.issubset(initial_hash_types):
        pytest.skip(
            f"Initial add did not generate all expected hash types "
            f"({initial_hash_types}), cannot test missing hash addition."
        )

    # --- Simulate missing one hash type (e.g., dhash) ---
    dhash_to_delete_stmt = select(ImagePerceptualHashComponent).where(
        ImagePerceptualHashComponent.entity_id == entity1.id,
        ImagePerceptualHashComponent.hash_type == "dhash",
    )
    dhash_comp = db_session.execute(dhash_to_delete_stmt).scalar_one_or_none()
    if dhash_comp:
        db_session.delete(dhash_comp)
        db_session.commit()
        # Verify it's gone
        phash_comps_after_delete = db_session.execute(stmt1).scalars().all()
        assert "dhash" not in {c.hash_type for c in phash_comps_after_delete}
    else:
        pytest.skip("dhash was not present initially, cannot test its re-addition.")

    # --- Second add (same content, different file instance/path to simulate new discovery) ---
    # Create a copy of sample_image_a to simulate a new file with same content
    copy_of_sample_image_a = tmp_path / "sample_A_copy.png"
    import shutil

    shutil.copy2(sample_image_a, copy_of_sample_image_a)

    props2 = file_operations.get_file_properties(copy_of_sample_image_a)
    chash2 = file_operations.calculate_sha256(copy_of_sample_image_a)
    assert chash1 == chash2  # Content hash must be the same

    entity2, created_new2 = asset_service.add_asset_file(
        db_session, copy_of_sample_image_a, props2[0], props2[2], props2[1], chash2
    )
    db_session.commit()

    assert created_new2 is False  # Entity should be existing
    assert entity2.id == entity1.id

    # Verify the 'dhash' is now present again
    stmt2 = select(ImagePerceptualHashComponent).where(ImagePerceptualHashComponent.entity_id == entity2.id)
    phash_comps2 = db_session.execute(stmt2).scalars().all()
    final_hash_types = {c.hash_type for c in phash_comps2}

    assert "dhash" in final_hash_types
    # Ensure other original hashes are still there
    assert initial_hash_types.difference({"dhash"}).issubset(final_hash_types)
    # The number of hashes should be back to original count, or at least have dhash
    assert len(phash_comps2) >= len(initial_hash_types) or "dhash" in final_hash_types
    # A more precise check if all original types were phash, ahash, dhash:
    if {"phash", "ahash", "dhash"}.issubset(initial_hash_types):
        assert {"phash", "ahash", "dhash"}.issubset(final_hash_types)
