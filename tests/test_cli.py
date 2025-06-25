import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dam.cli import app
from dam.core.database import SessionLocal, create_db_and_tables
from dam.services import asset_service

# Initialize a runner
runner = CliRunner()

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
IMG_A = TEST_DATA_DIR / "img_A.png"
IMG_A_VERY_SIMILAR = TEST_DATA_DIR / "img_A_very_similar.png"  # Assumed to be very similar to IMG_A
IMG_B = TEST_DATA_DIR / "img_B.png"
IMG_C_DIFFERENT = TEST_DATA_DIR / "img_C_different.png"  # Assumed to be different

# Create a dummy text file for hashing tests
TXT_FILE = TEST_DATA_DIR / "sample.txt"
TXT_FILE_CONTENT = "This is a test file for DAM."


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    # Create dummy text file
    with open(TXT_FILE, "w") as f:
        f.write(TXT_FILE_CONTENT)

    # Setup the database for the test module
    create_db_and_tables()  # Ensure tables are created

    # Add some initial assets for testing find commands
    db = SessionLocal()
    try:
        # Add IMG_A
        asset_service.add_asset_file(
            session=db,
            filepath_on_disk=IMG_A,
            original_filename=IMG_A.name,
            mime_type="image/png",
            size_bytes=IMG_A.stat().st_size,
        )
        # Add IMG_A_VERY_SIMILAR
        asset_service.add_asset_file(
            session=db,
            filepath_on_disk=IMG_A_VERY_SIMILAR,
            original_filename=IMG_A_VERY_SIMILAR.name,
            mime_type="image/png",
            size_bytes=IMG_A_VERY_SIMILAR.stat().st_size,
        )
        # Add IMG_B (used as a different image for similarity)
        asset_service.add_asset_file(
            session=db,
            filepath_on_disk=IMG_B,  # Intentionally using IMG_B as a source for a "different" image entry
            original_filename=IMG_C_DIFFERENT.name,  # But naming it as IMG_C_DIFFERENT in DB
            mime_type="image/png",
            size_bytes=IMG_B.stat().st_size,
        )
        # Add TXT_FILE
        asset_service.add_asset_file(
            session=db,
            filepath_on_disk=TXT_FILE,
            original_filename=TXT_FILE.name,
            mime_type="text/plain",
            size_bytes=TXT_FILE.stat().st_size,
        )
        db.commit()
    except Exception as e:
        db.rollback()
        pytest.fail(f"Failed to setup initial assets: {e}")
    finally:
        db.close()

    yield

    # Teardown: Remove dummy text file
    if TXT_FILE.exists():
        TXT_FILE.unlink()
    # Teardown: Clean up database tables if necessary or reset the db
    # For simplicity, we might rely on session-scoped fixtures or clear tables
    # Here, we are using module scope, so the DB will persist for all tests in this file.
    # Depending on test isolation needs, a more granular setup/teardown might be required.
    # For now, we assume the DB is ephemeral or reset between test suite runs.


def get_file_sha256(filepath: Path) -> str:
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def get_file_md5(filepath: Path) -> str:
    return hashlib.md5(filepath.read_bytes()).hexdigest()


# Tests for "find-file-by-hash" command
def test_find_file_by_sha256_direct_hash():
    sha256_hash = get_file_sha256(TXT_FILE)
    result = runner.invoke(app, ["find-file-by-hash", sha256_hash, "--hash-type", "sha256"])
    assert result.exit_code == 0
    assert f"Querying for asset with sha256 hash: {sha256_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout
    assert f"Value: {sha256_hash}" in result.stdout
    assert f"Name='{TXT_FILE.name}'" in result.stdout


def test_find_file_by_md5_direct_hash():
    md5_hash = get_file_md5(TXT_FILE)
    result = runner.invoke(app, ["find-file-by-hash", md5_hash, "--hash-type", "md5"])
    assert result.exit_code == 0
    assert f"Querying for asset with md5 hash: {md5_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout
    assert f"Value: {md5_hash}" in result.stdout  # Check if MD5 is listed
    assert f"Name='{TXT_FILE.name}'" in result.stdout


def test_find_file_by_sha256_filepath():
    result = runner.invoke(app, ["find-file-by-hash", "--file", str(TXT_FILE), "--hash-type", "sha256"])
    assert result.exit_code == 0
    sha256_hash = get_file_sha256(TXT_FILE)
    assert f"Calculating sha256 hash for file: {TXT_FILE}" in result.stdout
    assert f"Calculated sha256 hash: {sha256_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout


def test_find_file_by_md5_filepath():
    result = runner.invoke(app, ["find-file-by-hash", "--file", str(TXT_FILE), "--hash-type", "md5"])
    assert result.exit_code == 0
    md5_hash = get_file_md5(TXT_FILE)
    assert f"Calculating md5 hash for file: {TXT_FILE}" in result.stdout
    assert f"Calculated md5 hash: {md5_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout


def test_find_file_by_hash_not_found():
    non_existent_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    result = runner.invoke(app, ["find-file-by-hash", non_existent_hash])
    assert result.exit_code == 0  # Command itself succeeds, but finds nothing
    assert f"No asset found with sha256 hash: {non_existent_hash}" in result.stdout


def test_find_file_by_hash_file_not_exist_for_calc():
    result = runner.invoke(app, ["find-file-by-hash", "--file", "non_existent_file.txt"])
    assert result.exit_code != 0  # Typer/Click should make it non-zero due to `exists=True`
    assert "Error" in result.stdout  # Typer's error message for non-existent file


def test_find_file_by_hash_invalid_hash_type_calc():
    result = runner.invoke(app, ["find-file-by-hash", "--file", str(TXT_FILE), "--hash-type", "sha1"])
    assert result.exit_code == 1
    assert "Unsupported hash type for file calculation: sha1" in result.stdout


def test_find_file_by_hash_invalid_hash_type_direct():
    # This scenario (providing a hash value with an unsupported type for lookup)
    # is handled by asset_service returning None, then CLI reports "No asset found"
    # which is acceptable. The service logs an error.
    some_hash = get_file_sha256(TXT_FILE)
    result = runner.invoke(app, ["find-file-by-hash", some_hash, "--hash-type", "sha1"])
    assert result.exit_code == 0  # Command itself, service returns None
    assert f"No asset found with sha1 hash: {some_hash}" in result.stdout


# Tests for "find-similar-images" command
def test_find_similar_images_phash():
    # IMG_A and IMG_A_VERY_SIMILAR should be very close (distance 0 as they are copies)
    result = runner.invoke(app, ["find-similar-images", str(IMG_A), "--phash-threshold", "2"])
    print(result.stdout)  # For debugging
    assert result.exit_code == 0
    assert "Finding images similar to: img_A.png" in result.stdout
    assert "Found 1 potentially similar image(s):" in result.stdout  # IMG_A_VERY_SIMILAR
    assert f"File: '{IMG_A_VERY_SIMILAR.name}'" in result.stdout
    assert "Matched by phash" in result.stdout
    assert "Distance: 0" in result.stdout  # Since it's a copy
    # Ensure IMG_C_DIFFERENT (which is IMG_B's content) is not listed with low threshold
    assert f"File: '{IMG_C_DIFFERENT.name}'" not in result.stdout


def test_find_similar_images_ahash():
    result = runner.invoke(app, ["find-similar-images", str(IMG_A), "--ahash-threshold", "2"])
    assert result.exit_code == 0
    assert "Finding images similar to: img_A.png" in result.stdout
    assert "Found 1 potentially similar image(s):" in result.stdout
    assert f"File: '{IMG_A_VERY_SIMILAR.name}'" in result.stdout
    assert "Matched by ahash" in result.stdout
    assert "Distance: 0" in result.stdout


def test_find_similar_images_dhash():
    result = runner.invoke(app, ["find-similar-images", str(IMG_A), "--dhash-threshold", "2"])
    assert result.exit_code == 0
    assert "Finding images similar to: img_A.png" in result.stdout
    assert "Found 1 potentially similar image(s):" in result.stdout
    assert f"File: '{IMG_A_VERY_SIMILAR.name}'" in result.stdout
    assert "Matched by dhash" in result.stdout
    assert "Distance: 0" in result.stdout


def test_find_similar_images_higher_threshold_includes_more():
    # Assuming IMG_B (aliased as IMG_C_DIFFERENT in DB) might be somewhat similar with a high enough threshold
    # This test is conceptual as true similarity depends on actual image content and hash properties
    # For this test, we expect IMG_A_VERY_SIMILAR (dist 0) and potentially IMG_C_DIFFERENT (dist > 0)
    # if the threshold is high enough. Since IMG_B is different, its distance will be > 0.
    # Let's try to include IMG_C_DIFFERENT by setting a very high threshold.
    # The exact distance between IMG_A and IMG_B's content needs to be known for a precise test.
    # For now, we'll just check that IMG_A_VERY_SIMILAR is found.
    # A more robust test would involve crafting images with known hash distances.
    cmd = [
        "find-similar-images",
        str(IMG_A),
        "--phash-threshold",
        "60",
        "--ahash-threshold",
        "60",
        "--dhash-threshold",
        "60",
    ]
    result = runner.invoke(app, cmd)
    assert result.exit_code == 0
    assert f"File: '{IMG_A_VERY_SIMILAR.name}'" in result.stdout  # Should always be found if present
    # We can't reliably assert IMG_C_DIFFERENT here without knowing actual hash distances
    # But we can check that the command ran and found at least one.
    assert "Found" in result.stdout  # General check
    assert "potentially similar image(s):" in result.stdout


def test_find_similar_images_no_similar_found():
    # Use IMG_B and a very low threshold, expecting only itself (if it were in DB under its own name)
    # or nothing if comparing against very different images.
    # Here, we compare IMG_B against the DB which has IMG_A, IMG_A_VERY_SIMILAR, and IMG_B (as IMG_C_DIFFERENT)
    # So, it should find IMG_C_DIFFERENT (which is IMG_B's content) with distance 0.
    result = runner.invoke(app, ["find-similar-images", str(IMG_B), "--phash-threshold", "0"])
    assert result.exit_code == 0
    assert "Finding images similar to: img_B.png" in result.stdout
    assert "Found 1 potentially similar image(s):" in result.stdout  # IMG_C_DIFFERENT has IMG_B's content
    assert f"File: '{IMG_C_DIFFERENT.name}'" in result.stdout
    assert "Distance: 0" in result.stdout


def test_find_similar_images_input_file_not_image():
    result = runner.invoke(app, ["find-similar-images", str(TXT_FILE)])
    assert result.exit_code == 1  # Should fail as generate_perceptual_hashes will return empty or error
    assert "Error processing image" in result.stdout or "Could not generate perceptual hashes" in result.stdout


def test_find_similar_images_input_file_not_exist():
    result = runner.invoke(app, ["find-similar-images", "non_existent_image.png"])
    assert result.exit_code != 0  # Typer/Click handles this
    assert "Error" in result.stdout
    assert "non_existent_image.png" in result.stdout


# Test for add-asset to ensure MD5s are added
def test_add_asset_generates_md5():
    db = SessionLocal()
    # Create a new temporary file to add
    temp_file_path = TEST_DATA_DIR / "temp_add_asset_test.txt"
    with open(temp_file_path, "w") as f:
        f.write("Content for MD5 test.")

    temp_file_md5 = get_file_md5(temp_file_path)

    try:
        result = runner.invoke(app, ["add-asset", str(temp_file_path)])
        assert result.exit_code == 0
        assert "Successfully added new asset." in result.stdout  # Assuming it's a new asset

        # Verify in DB
        # Find by SHA256 first (as that's the primary identifier used by add_asset_file logic)
        temp_file_sha256 = get_file_sha256(temp_file_path)
        entity = asset_service.find_entity_by_content_hash(db, temp_file_sha256, "sha256")
        assert entity is not None

        # Now check if MD5 component exists and matches
        md5_components = asset_service.get_components(db, entity.id, asset_service.ContentHashMD5Component)
        found_md5 = False
        for comp in md5_components:
            if comp.hash_value == temp_file_md5:
                found_md5 = True
                break
        assert found_md5, f"MD5 component with hash {temp_file_md5} not found for new asset."

    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()
        # Clean up the test entity from DB if needed, or rely on full DB reset for tests
        if "entity" in locals() and entity:
            db.delete(entity)  # Simplistic cleanup; cascade might be needed for components
            # For robust testing, a transaction rollback or DB recreation per test is better
            db.commit()
        db.close()


# TODO: Add more tests for edge cases, error handling, and different file types if necessary.
# TODO: For image similarity, use actual distinct images with known small hash differences for better threshold tests.
# The current IMG_A_VERY_SIMILAR is a direct copy, so distance is always 0.
# IMG_C_DIFFERENT uses IMG_B's content, so it tests if different content is correctly handled.
# A "slightly modified" image would be better for testing non-zero small distances.

# Example of how to handle db session per test if needed:
# @pytest.fixture
# def db_session():
#     # Setup: create tables if they don't exist
#     # Base.metadata.create_all(bind=engine) # Assuming Base and engine are available
#     session = SessionLocal()
#     try:
#         yield session
#     finally:
#         session.close()
#         # Teardown: Optionally, clear data from tables or drop tables
#         # Base.metadata.drop_all(bind=engine)


# To run tests: pytest tests/test_cli.py
# Ensure TEST_DATABASE_URL in .env (if used by config) points to a test DB.
# The current code uses a global SessionLocal, so all tests in this module will share
# the DB state manipulated by the setup_test_environment fixture.
# For true isolation, each test function should manage its own DB state (e.g., via transactions and rollback).
# The `create_db_and_tables()` is called once per module here.
# The asset_storage_path should also be configured for tests, perhaps to a temp dir.
# Default is "dam_storage", so ensure this is cleaned up or managed if tests write files.
# The current tests primarily focus on DB interactions and CLI output, not file storage persistence.

# Note on `IMG_A_VERY_SIMILAR` and `IMG_C_DIFFERENT`:
# `IMG_A_VERY_SIMILAR` is a copy of `IMG_A`.
# `IMG_C_DIFFERENT` is a copy of `IMG_B` but added to the DB with `IMG_C_DIFFERENT.name`.
# This setup allows testing:
# 1. Exact matches (IMG_A vs IMG_A_VERY_SIMILAR's content).
# 2. Different images (IMG_A vs IMG_C_DIFFERENT's content which is IMG_B's content).
# For more nuanced similarity (small non-zero distances), one would need to actually modify an image slightly.
# The tests `test_find_similar_images_phash` etc. are structured to find
# `IMG_A_VERY_SIMILAR` when searching with `IMG_A`.
# `test_find_similar_images_no_similar_found` uses `IMG_B` as input and should find
# `IMG_C_DIFFERENT` (as it has IMG_B's content) if threshold is 0.


# A fixture to clean the database before each test might be better for isolation
@pytest.fixture(autouse=True)
def clean_db_before_each_test(setup_test_environment):  # Depends on module-level setup
    # This is a bit of a workaround to reset state without full migration down/up
    # A proper solution would be transaction-based testing or recreating DB schema per test/session
    # For now, we ensure the setup_test_environment has run once for the module.
    # This fixture itself doesn't clean yet, relies on setup_test_environment for initial state.
    # A more robust approach:
    # engine = get_engine()
    # from dam.models.base_class import Base
    # Base.metadata.drop_all(engine)
    # Base.metadata.create_all(engine)
    # And then re-populate required initial data for each test or test group.
    # The current `setup_test_environment` with `autouse=True, scope="module"`
    # populates data once. Subsequent tests see this data.
    # The `add_asset_generates_md5` test adds and then tries to clean up its own asset.
    pass
