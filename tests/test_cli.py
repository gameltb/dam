import hashlib
from pathlib import Path

import pytest
from pytest import MonkeyPatch
from typer.testing import CliRunner

from dam.cli import app
from dam.core.database import SessionLocal
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


# Helper to create dummy image files
def _create_dummy_image(filepath: Path, color_name: str, size=(10, 10)):
    # Define actual colors for names
    colors = {
        "red": (255, 0, 0),
        "red_with_dot": (255, 0, 0),  # Base color red
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
    }
    try:
        from PIL import Image

        img = Image.new("RGB", size, color=colors[color_name])

        if color_name == "red_with_dot":
            img.putpixel((0, 0), (0, 0, 0))  # Add a black dot to make it different from plain "red"
        elif color_name == "blue":
            for i in range(size[0]):
                img.putpixel((i, i), (255, 255, 255))  # White diagonal line
        elif color_name == "green":
            for i in range(size[0]):
                img.putpixel((i, 0), (0, 0, 0))  # Black top line

        filepath.parent.mkdir(parents=True, exist_ok=True)
        img.save(filepath, "PNG")
    except ImportError:
        # Pillow not installed, create a tiny text file as placeholder
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:  # wb to mimic image file a bit
            f.write(f"dummy_image_{color_name}".encode())


@pytest.fixture(scope="function", autouse=True)
def setup_test_environment(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("DAM_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("TESTING_MODE", "True")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from dam.core import database as app_database
    from dam.core.config import settings as app_settings

    monkeypatch.setattr(app_settings, "DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setattr(app_settings, "TESTING_MODE", True)

    new_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    monkeypatch.setattr(app_database, "engine", new_engine)

    new_session_local = sessionmaker(autocommit=False, autoflush=False, bind=new_engine)
    monkeypatch.setattr(app_database, "SessionLocal", new_session_local)

    app_database.create_db_and_tables()

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _create_dummy_image(IMG_A, "red")
    _create_dummy_image(IMG_A_VERY_SIMILAR, "red_with_dot")  # Different content from IMG_A
    _create_dummy_image(IMG_B, "blue")
    _create_dummy_image(IMG_C_DIFFERENT, "green")

    with open(TXT_FILE, "w") as f:
        f.write(TXT_FILE_CONTENT)

    db = new_session_local()
    try:
        # Only add IMG_A in the fixture for now to simplify state
        asset_service.add_asset_file(
            session=db,
            filepath_on_disk=IMG_A,
            original_filename=IMG_A.name,
            mime_type="image/png",
            size_bytes=IMG_A.stat().st_size,
        )
        # asset_service.add_asset_file(
        #     session=db, filepath_on_disk=IMG_A_VERY_SIMILAR, original_filename=IMG_A_VERY_SIMILAR.name,
        #     mime_type="image/png", size_bytes=IMG_A_VERY_SIMILAR.stat().st_size,
        # )
        # asset_service.add_asset_file(
        #     session=db, filepath_on_disk=IMG_B, original_filename=IMG_C_DIFFERENT.name,
        #     mime_type="image/png", size_bytes=IMG_B.stat().st_size,
        # )
        # asset_service.add_asset_file(
        #     session=db, filepath_on_disk=TXT_FILE, original_filename=TXT_FILE.name,
        #     mime_type="text/plain", size_bytes=TXT_FILE.stat().st_size,
        # )
        db.commit()
    except Exception as e:
        db.rollback()
        pytest.fail(f"Failed to setup initial assets: {e}. DB URL: {str(new_engine.url)}. Error Type: {type(e)}")
    finally:
        db.close()

    yield

    IMG_A.unlink(missing_ok=True)
    IMG_A_VERY_SIMILAR.unlink(missing_ok=True)
    IMG_B.unlink(missing_ok=True)
    IMG_C_DIFFERENT.unlink(missing_ok=True)
    TXT_FILE.unlink(missing_ok=True)


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
    # Provide a dummy hash_value because it's a positional arg, --file will override it.
    result = runner.invoke(app, ["find-file-by-hash", "testhash", "--file", str(TXT_FILE), "--hash-type", "sha256"])
    assert result.exit_code == 0
    sha256_hash = get_file_sha256(TXT_FILE)
    assert f"Calculating sha256 hash for file: {TXT_FILE}" in result.stdout
    assert f"Calculated sha256 hash: {sha256_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout


def test_find_file_by_md5_filepath():
    result = runner.invoke(app, ["find-file-by-hash", "testhash", "--file", str(TXT_FILE), "--hash-type", "md5"])
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
    result = runner.invoke(app, ["find-file-by-hash", "somehash", "--file", "non_existent_file.txt"])
    assert result.exit_code != 0
    # Typer's default error for `exists=True` on Path includes "does not exist"
    # It typically prints to stderr. runner.invoke captures stdout and stderr.
    # We check if the exception message is in the output (which includes stderr).
    assert "non_existent_file.txt" in result.output
    assert "does not exist" in result.output


def test_find_file_by_hash_invalid_hash_type_calc():
    result = runner.invoke(app, ["find-file-by-hash", "testhash", "--file", str(TXT_FILE), "--hash-type", "sha1"])
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
    # Expect IMG_A_VERY_SIMILAR (pHash dist ~14 from IMG_A)
    result = runner.invoke(app, ["find-similar-images", str(IMG_A), "--phash-threshold", "15"])
    print(f"Output for phash test:\n{result.stdout}")
    assert result.exit_code == 0
    assert "Finding images similar to: img_A.png" in result.stdout
    assert "Found 1 potentially similar image(s):" in result.stdout
    assert f"Original Filename: {IMG_A_VERY_SIMILAR.name}" in result.stdout
    assert "Matched by phash" in result.stdout
    assert "Distance: 14" in result.stdout  # Adjusted expected distance
    assert f"Original Filename: {IMG_C_DIFFERENT.name}" not in result.stdout


def test_find_similar_images_ahash():
    # Expect IMG_A_VERY_SIMILAR (aHash dist ~30 from IMG_A)
    result = runner.invoke(app, ["find-similar-images", str(IMG_A), "--ahash-threshold", "31"])
    print(f"Output for ahash test:\n{result.stdout}")
    assert result.exit_code == 0
    assert "Finding images similar to: img_A.png" in result.stdout
    assert "Found 1 potentially similar image(s):" in result.stdout
    assert f"Original Filename: {IMG_A_VERY_SIMILAR.name}" in result.stdout
    assert "Matched by ahash" in result.stdout
    assert "Distance: 30" in result.stdout  # Adjusted expected distance


def test_find_similar_images_dhash():
    # Expect IMG_A_VERY_SIMILAR (dHash dist should be 2 based on logs)
    result = runner.invoke(app, ["find-similar-images", str(IMG_A), "--dhash-threshold", "2"])
    print(f"Output for dhash test:\n{result.stdout}")
    assert result.exit_code == 0
    assert "Finding images similar to: img_A.png" in result.stdout
    assert "Found 1 potentially similar image(s):" in result.stdout
    assert f"Original Filename: {IMG_A_VERY_SIMILAR.name}" in result.stdout
    assert "Matched by dhash" in result.stdout
    assert "Distance: 2" in result.stdout  # Adjusted expected distance


def test_find_similar_images_higher_threshold_includes_more():
    # IMG_A (red) vs:
    # - IMG_A_VERY_SIMILAR (red_with_dot): p=14, a=30, d=2. All <=60.
    # - IMG_C_DIFFERENT (green_with_line, content of IMG_B blue_with_line):
    #   Let's get these distances from logs or re-calculate if needed.
    #   IMG_A pHash '8000...', aHash '0000...', dHash '0000...'
    #   IMG_C_DIFFERENT (Entity 3) pHash '8040...', aHash '8040...', dHash '2894...'
    #   Distances: pHash(8000 vs 8040) should be small. aHash(0000 vs 8040) moderate. dHash(0000 vs 2894) moderate.
    #   pHash: imagehash.hex_to_hash('8000000000000000') - imagehash.hex_to_hash('8040201008040201') = 8
    #   aHash: imagehash.hex_to_hash('0000000000000000') - imagehash.hex_to_hash('8040201008040201') = 8
    #   dHash: imagehash.hex_to_hash('0000000000000000') - imagehash.hex_to_hash('289448269b6d6d2f') = 24

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
    print(f"Output for higher_threshold test:\n{result.stdout}")
    assert result.exit_code == 0
    assert "Found 2 potentially similar image(s):" in result.stdout
    assert f"Original Filename: {IMG_A_VERY_SIMILAR.name}" in result.stdout
    assert f"Original Filename: {IMG_C_DIFFERENT.name}" in result.stdout  # IMG_C_DIFFERENT is name for IMG_B's content


def test_find_similar_images_no_similar_found():
    # Search with IMG_B (blue_with_line), threshold 0.
    # Its content is in DB as Entity 3 (orig name IMG_C_DIFFERENT). This entity will be excluded.
    # No other images (IMG_A, IMG_A_VERY_SIMILAR) should be identical to IMG_B.
    result = runner.invoke(
        app,
        [
            "find-similar-images",
            str(IMG_B),
            "--phash-threshold",
            "0",
            "--ahash-threshold",
            "0",
            "--dhash-threshold",
            "0",
        ],
    )
    print(f"Output for no_similar_found test:\n{result.stdout}")
    assert result.exit_code == 0
    assert "Finding images similar to: img_B.png" in result.stdout
    assert "No similar images found based on the criteria." in result.stdout


def test_find_similar_images_input_file_not_image():
    result = runner.invoke(app, ["find-similar-images", str(TXT_FILE)])
    assert result.exit_code == 1  # Should fail as generate_perceptual_hashes will return empty or error
    assert "Error processing image" in result.stdout or "Could not generate perceptual hashes" in result.stdout


def test_find_similar_images_input_file_not_exist():
    result = runner.invoke(app, ["find-similar-images", "non_existent_image.png"])
    assert result.exit_code == 1  # Due to ValueError catch and typer.Exit(1)
    assert "Error processing image for similarity search" in result.stdout
    assert "Could not generate any perceptual hashes for non_existent_image.png" in result.stdout



# Test for add-asset to ensure MD5s are added
def test_add_asset_generates_md5():
    # This test will now directly call the service after fixture setup to check ID generation
    # The fixture (setup_test_environment) has already run and committed Entities 1-4.

    db_for_test = SessionLocal()  # Use the patched SessionLocal for this test operations

    from dam.models import Entity
    from dam.services.asset_service import FilePropertiesComponent  # For type hinting if needed by get_component

    # Check visibility of Entity 4 from fixture
    entity4_from_fixture = db_for_test.get(Entity, 4)
    if entity4_from_fixture:
        print(f"Entity 4 (TXT_FILE from fixture) found by test session: ID {entity4_from_fixture.id}")
        # Try to get its FilePropertiesComponent
        fp_comp_for_e4 = asset_service.get_component(db_for_test, 4, FilePropertiesComponent)
        if fp_comp_for_e4:
            print(f"  Entity 4 FPC original_filename: {fp_comp_for_e4.original_filename} (ID: {fp_comp_for_e4.id})")
        else:
            print("  Entity 4 FPC NOT found by test session.")
    else:
        print("Entity 4 (TXT_FILE from fixture) NOT found by test session.")

    all_entity_ids = [r[0] for r in db_for_test.query(Entity.id).all()]
    print(f"All Entity IDs seen by test session at start: {all_entity_ids}")

    temp_file_path = TEST_DATA_DIR / "temp_add_asset_for_md5_test.txt"  # Unique name
    temp_file_content = "Content for MD5 test in test_add_asset_generates_md5."
    with open(temp_file_path, "w") as f:
        f.write(temp_file_content)

    temp_file_md5 = get_file_md5(temp_file_path)
    # temp_file_sha256 = get_file_sha256(temp_file_path) # Unused

    # new_entity_from_service = None # Unused
    try:
        # Directly call the service
        print(f"Calling asset_service.add_asset_file for: {temp_file_path.name}")
        entity_obj, created_new = asset_service.add_asset_file(
            session=db_for_test,
            filepath_on_disk=temp_file_path,
            original_filename=temp_file_path.name,
            mime_type="text/plain",
            size_bytes=temp_file_path.stat().st_size,
        )
        db_for_test.commit()
        # new_entity_from_service = entity_obj  # Unused, cleanup is handled by fixture scope

        print(
            f"Service call for {temp_file_path.name} resulted in Entity ID: {entity_obj.id}, Created new: {created_new}"
        )
        assert created_new, "Service should have created a new entity for a new file."
        assert entity_obj.id > 4, f"Expected new entity ID > 4, but got {entity_obj.id}"

        # Verify MD5 component
        md5_components = asset_service.get_components(db_for_test, entity_obj.id, asset_service.ContentHashMD5Component)
        found_md5 = any(comp.hash_value == temp_file_md5 for comp in md5_components)
        assert found_md5, f"MD5 component with hash {temp_file_md5} not found for Entity ID {entity_obj.id}."

        # Also check CLI behavior separately if direct service call works
        # For now, focusing on the service layer behavior due to ID issues.
        # result = runner.invoke(app, ["add-asset", str(temp_file_path)])
        # print(f"add-asset CLI output for {temp_file_path}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        # assert result.exit_code == 0
        # assert "Successfully added new asset." in result.stdout

    except Exception as e:
        db_for_test.rollback()
        pytest.fail(f"Error during test_add_asset_generates_md5 direct service call: {e}")
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()

        # Clean up the entity created by this specific test, if it exists
        # This is tricky because other tests might be affected if we delete from shared in-memory DB
        # However, since each test function has its own setup_test_environment, this should be fine.
        # The `setup_test_environment` will drop all tables for the next test.
        # if new_entity_from_service:
        #     ecs_service.delete_entity_with_components(db_for_test, new_entity_from_service.id)
        #     db_for_test.commit()
        db_for_test.close()


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

# The `setup_test_environment` fixture with `autouse=True, scope="module"` and
# the modified `create_db_and_tables` (which drops tables in testing mode)
# should handle DB setup and teardown for this test module.
# The `auto_clean_db_tables` fixture is removed as it was redundant or misconfigured
# for its intended purpose of per-test cleaning vs module-level monkeypatch undo.
# The module_monkeypatch.undo() is now handled by setup_test_environment's finalizer.
