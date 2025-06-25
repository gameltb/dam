import hashlib
from pathlib import Path

import pytest
from pytest import MonkeyPatch
from typer.testing import CliRunner

# App is imported within a fixture to ensure patches are applied first
from dam.core.database import SessionLocal  # Keep for fixture DB ops if needed by setup
from dam.services import asset_service

# Initialize a runner
runner = CliRunner()


@pytest.fixture
def test_app(setup_test_environment):  # Depends on setup_test_environment having run
    """Provides the Typer app instance after environment setup and patching."""
    from dam.cli import app as actual_app  # Import app after patches are applied

    return actual_app


# Test data directory (relative to this test file)
TEST_DATA_DIR = Path(__file__).parent / "test_data"
IMG_A_FILENAME = "img_A.png"
IMG_A_VERY_SIMILAR_FILENAME = "img_A_very_similar.png"
IMG_B_FILENAME = "img_B.png"
IMG_C_DIFFERENT_FILENAME = "img_C_different.png"  # Asset named this will use IMG_B's content
TXT_FILENAME = "sample.txt"
TXT_FILE_CONTENT = "This is a test file for DAM."

# Define Paths for physical file creation by the fixture
# These will be created under tmp_path by the fixture for isolation
# The names here are just for clarity; the fixture uses tmp_path.
IMG_A_SOURCE_PATH = TEST_DATA_DIR / IMG_A_FILENAME
IMG_A_VERY_SIMILAR_SOURCE_PATH = TEST_DATA_DIR / IMG_A_VERY_SIMILAR_FILENAME
IMG_B_SOURCE_PATH = TEST_DATA_DIR / IMG_B_FILENAME
# IMG_C_DIFFERENT_SOURCE_PATH is not strictly needed as its content comes from IMG_B
TXT_FILE_SOURCE_PATH = TEST_DATA_DIR / TXT_FILENAME


# Helper to create dummy image files
def _create_dummy_image(filepath: Path, color_name: str, size=(10, 10)):
    colors = {
        "red": (255, 0, 0),
        "red_with_dot": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
    }
    try:
        from PIL import Image

        img = Image.new("RGB", size, color=colors[color_name])
        if color_name == "red_with_dot":
            img.putpixel((0, 0), (0, 0, 0))
        elif color_name == "blue":
            for i in range(size[0]):
                img.putpixel((i, i), (255, 255, 255))
        elif color_name == "green":  # Make green distinct from red and blue
            # Fill with green
            for x_coord in range(size[0]):  # Black horizontal line
                img.putpixel((x_coord, size[1] // 2), (0, 0, 0))
            img.putpixel((1, 1), (255, 0, 0))  # Add a red pixel
            img.putpixel((size[0] - 2, size[1] - 2), (0, 0, 255))  # Add a blue pixel

        filepath.parent.mkdir(parents=True, exist_ok=True)
        img.save(filepath, "PNG")
    except ImportError:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(f"dummy_image_{color_name}".encode())


@pytest.fixture(scope="function", autouse=True)
def setup_test_environment(monkeypatch: MonkeyPatch, tmp_path: Path):
    # 1. Configure temporary paths for assets and DB
    temp_storage_path = tmp_path / "dam_cli_asset_storage"
    temp_storage_path.mkdir(parents=True, exist_ok=True)

    db_file = tmp_path / "test_cli_dam.db"
    test_db_url = f"sqlite:///{db_file}"

    # 2. Set environment variables (Pydantic settings will pick these up)
    monkeypatch.setenv("DAM_ASSET_STORAGE_PATH", str(temp_storage_path))
    monkeypatch.setenv("DAM_DATABASE_URL", test_db_url)
    monkeypatch.setenv("TESTING_MODE", "True")

    # 3. Import settings and core database module AFTER env vars are set
    #    and monkeypatch the live settings instance for good measure.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from dam.core.config import settings as app_settings  # Reloads settings based on new env vars
    from dam.core import database as app_database

    monkeypatch.setattr(app_settings, "ASSET_STORAGE_PATH", str(temp_storage_path))
    monkeypatch.setattr(app_settings, "DATABASE_URL", test_db_url)
    monkeypatch.setattr(app_settings, "TESTING_MODE", True)

    # 4. Create a new engine with the temporary DB URL and patch it into app_database
    new_engine = create_engine(app_settings.DATABASE_URL, connect_args={"check_same_thread": False})
    monkeypatch.setattr(app_database, "engine", new_engine)

    # 5. Create a new SessionLocal bound to the new engine and patch it
    new_session_local = sessionmaker(autocommit=False, autoflush=False, bind=new_engine)
    monkeypatch.setattr(app_database, "SessionLocal", new_session_local)

    # 6. Ensure all models are loaded into Base.metadata before creating tables
    import dam.models  # noqa: F401 to ensure models are registered

    app_database.create_db_and_tables()  # Uses the patched app_database.engine

    # 7. Create physical test files in a temporary "source" location (within tmp_path)
    #    These are the files that will be "added" to the DAM.
    #    The DAM itself will store them in `temp_storage_path` based on content.
    source_files_dir = tmp_path / "source_files"
    source_files_dir.mkdir(exist_ok=True)

    img_a_path = source_files_dir / IMG_A_FILENAME
    img_a_similar_path = source_files_dir / IMG_A_VERY_SIMILAR_FILENAME
    img_b_path = source_files_dir / IMG_B_FILENAME
    img_c_path = source_files_dir / IMG_C_DIFFERENT_FILENAME  # For asset named IMG_C_DIFFERENT
    txt_file_path = source_files_dir / TXT_FILENAME

    _create_dummy_image(img_a_path, "red")
    _create_dummy_image(img_a_similar_path, "red_with_dot")
    _create_dummy_image(img_b_path, "blue")
    _create_dummy_image(img_c_path, "green")  # Give IMG_C different content from IMG_B
    # for tests like find_similar_images_higher_threshold_includes_more
    # where IMG_C_DIFFERENT asset uses IMG_B's content.
    # This img_c_path (green) is for distinct content.

    with open(txt_file_path, "w") as f:
        f.write(TXT_FILE_CONTENT)

    # Global paths for tests to refer to these source files
    global _FIXTURE_IMG_A, _FIXTURE_IMG_A_SIMILAR, _FIXTURE_IMG_B, _FIXTURE_IMG_C_GREEN, _FIXTURE_TXT_FILE
    _FIXTURE_IMG_A = img_a_path
    _FIXTURE_IMG_A_SIMILAR = img_a_similar_path
    _FIXTURE_IMG_B = img_b_path  # Blue content
    _FIXTURE_IMG_C_GREEN = img_c_path  # Green content
    _FIXTURE_TXT_FILE = txt_file_path

    # 8. Add assets to the temporary database
    db = new_session_local()  # Use the patched SessionLocal
    try:
        asset_service.add_asset_file(
            session=db,
            filepath_on_disk=_FIXTURE_IMG_A,
            original_filename=IMG_A_FILENAME,
            mime_type="image/png",
            size_bytes=_FIXTURE_IMG_A.stat().st_size,
        )
        asset_service.add_asset_file(
            session=db,
            filepath_on_disk=_FIXTURE_IMG_A_SIMILAR,
            original_filename=IMG_A_VERY_SIMILAR_FILENAME,
            mime_type="image/png",
            size_bytes=_FIXTURE_IMG_A_SIMILAR.stat().st_size,
        )
        # Asset named "img_C_different.png" will use content of _FIXTURE_IMG_B (blue image)
        asset_service.add_asset_file(
            session=db,
            filepath_on_disk=_FIXTURE_IMG_B,
            original_filename=IMG_C_DIFFERENT_FILENAME,
            mime_type="image/png",
            size_bytes=_FIXTURE_IMG_B.stat().st_size,
        )
        asset_service.add_asset_file(
            session=db,
            filepath_on_disk=_FIXTURE_TXT_FILE,
            original_filename=TXT_FILENAME,
            mime_type="text/plain",
            size_bytes=_FIXTURE_TXT_FILE.stat().st_size,
        )
        # Add the distinct green image as well, perhaps under its own name or a generic one
        asset_service.add_asset_file(
            session=db,
            filepath_on_disk=_FIXTURE_IMG_C_GREEN,
            original_filename="img_green_distinct.png",
            mime_type="image/png",
            size_bytes=_FIXTURE_IMG_C_GREEN.stat().st_size,
        )

        db.commit()
    except Exception as e:
        db.rollback()
        pytest.fail(
            f"Failed to setup initial assets in temporary DB: {e}. DB URL: {app_settings.DATABASE_URL}. Error: {type(e).__name__}"
        )
    finally:
        db.close()

    yield  # Test runs here

    # No need to manually delete files in tmp_path, pytest handles it.


def get_file_sha256(filepath: Path) -> str:
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def get_file_md5(filepath: Path) -> str:
    return hashlib.md5(filepath.read_bytes()).hexdigest()


# Tests for "find-file-by-hash" command
def test_find_file_by_sha256_direct_hash(test_app):
    sha256_hash = get_file_sha256(_FIXTURE_TXT_FILE)  # Use fixture path
    result = runner.invoke(test_app, ["find-file-by-hash", sha256_hash, "--hash-type", "sha256"])
    if result.exit_code != 0:
        print(f"CLI Error Output for test_find_file_by_sha256_direct_hash:\n{result.output}")
    assert result.exit_code == 0
    assert f"Querying for asset with sha256 hash: {sha256_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout
    assert f"Value: {sha256_hash}" in result.stdout
    assert f"Name='{TXT_FILENAME}'" in result.stdout


def test_find_file_by_md5_direct_hash(test_app):
    md5_hash = get_file_md5(_FIXTURE_TXT_FILE)  # Use fixture path
    result = runner.invoke(test_app, ["find-file-by-hash", md5_hash, "--hash-type", "md5"])
    assert result.exit_code == 0
    assert f"Querying for asset with md5 hash: {md5_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout
    assert f"Value: {md5_hash}" in result.stdout
    assert f"Name='{TXT_FILENAME}'" in result.stdout


def test_find_file_by_sha256_filepath(test_app):
    result = runner.invoke(
        test_app, ["find-file-by-hash", "testhash", "--file", str(_FIXTURE_TXT_FILE), "--hash-type", "sha256"]
    )  # Use fixture path
    assert result.exit_code == 0
    sha256_hash = get_file_sha256(_FIXTURE_TXT_FILE)
    assert f"Calculating sha256 hash for file: {str(_FIXTURE_TXT_FILE)}" in result.stdout
    assert f"Calculated sha256 hash: {sha256_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout


def test_find_file_by_md5_filepath(test_app):
    result = runner.invoke(
        test_app, ["find-file-by-hash", "testhash", "--file", str(_FIXTURE_TXT_FILE), "--hash-type", "md5"]
    )  # Use fixture path
    assert result.exit_code == 0
    md5_hash = get_file_md5(_FIXTURE_TXT_FILE)
    assert f"Calculating md5 hash for file: {str(_FIXTURE_TXT_FILE)}" in result.stdout
    assert f"Calculated md5 hash: {md5_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout


def test_find_file_by_hash_not_found(test_app):
    non_existent_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    result = runner.invoke(test_app, ["find-file-by-hash", non_existent_hash])
    assert result.exit_code == 0
    assert f"No asset found with sha256 hash: {non_existent_hash}" in result.stdout


def test_find_file_by_hash_file_not_exist_for_calc(test_app):
    result = runner.invoke(test_app, ["find-file-by-hash", "somehash", "--file", "non_existent_file.txt"])
    assert result.exit_code != 0
    assert "non_existent_file.txt" in result.output
    assert "does not exist" in result.output


def test_find_file_by_hash_invalid_hash_type_calc(test_app):
    result = runner.invoke(
        test_app, ["find-file-by-hash", "testhash", "--file", str(_FIXTURE_TXT_FILE), "--hash-type", "sha1"]
    )  # Use fixture path
    assert result.exit_code == 1
    assert "Unsupported hash type for file calculation: sha1" in result.stdout


def test_find_file_by_hash_invalid_hash_type_direct(test_app):
    some_hash = get_file_sha256(_FIXTURE_TXT_FILE)  # Use fixture path
    result = runner.invoke(test_app, ["find-file-by-hash", some_hash, "--hash-type", "sha1"])
    assert result.exit_code == 0
    assert f"No asset found with sha1 hash: {some_hash}" in result.stdout


# Tests for "find-similar-images" command
def test_find_similar_images_phash(test_app):
    result = runner.invoke(test_app, ["find-similar-images", str(_FIXTURE_IMG_A), "--phash-threshold", "15"])
    if result.exit_code != 0:
        print(f"Output for phash test (raw):\n{result.output}")  # Keep this for raw output
    print(f"Output for phash test (stdout):\n{result.stdout}")  # Print stdout specifically
    assert result.exit_code == 0
    assert f"Finding images similar to: {IMG_A_FILENAME}" in result.stdout
    assert "Found 2 potentially similar image(s):" in result.stdout
    # Check for the presence of the filenames in the output, using the actual CLI output format
    assert f"File: '{IMG_A_VERY_SIMILAR_FILENAME}'" in result.stdout  # Match on dHash
    assert f"File: '{IMG_C_DIFFERENT_FILENAME}'" in result.stdout  # Match on pHash
    assert f"Original Filename: img_green_distinct.png" not in result.stdout  # No match (also check File: '... format)
    assert f"File: 'img_green_distinct.png'" not in result.stdout

    # Ensure "Matched by phash" is present for one and "Matched by dhash" for another
    # More robustly, check that the correct entity is matched by the correct hash type
    # The output format is: Entity ID: X (Matched by HASH_TYPE, Distance: D)
    #                       File: 'FILENAME'

    # Check for IMG_A_VERY_SIMILAR (Entity 2) details
    assert f"Entity ID: 2 (Matched by dhash, Distance: 3)" in result.stdout
    assert f"File: '{IMG_A_VERY_SIMILAR_FILENAME}'" in result.stdout  # Redundant if the block is structured, but safe

    # Check for IMG_C_DIFFERENT (Entity 3) details
    assert f"Entity ID: 3 (Matched by phash, Distance: 7)" in result.stdout
    assert f"File: '{IMG_C_DIFFERENT_FILENAME}'" in result.stdout  # Redundant if the block is structured, but safe


def test_find_similar_images_ahash(test_app):
    # Set strict thresholds for pHash and dHash to isolate aHash match
    result = runner.invoke(
        test_app,
        [
            "find-similar-images",
            str(_FIXTURE_IMG_A),
            "--phash-threshold",
            "0",  # IMG_A_SIMILAR pHash dist is 24
            "--ahash-threshold",
            "31",  # IMG_A_SIMILAR aHash dist is 30 (match)
            "--dhash-threshold",
            "0",  # IMG_A_SIMILAR dHash dist is 3
        ],
    )
    if result.exit_code != 0:
        print(f"Output for ahash test (raw):\n{result.output}")
    print(f"Output for ahash test (stdout):\n{result.stdout}")
    assert result.exit_code == 0
    assert f"Finding images similar to: {IMG_A_FILENAME}" in result.stdout
    assert "Found 1 potentially similar image(s):" in result.stdout
    assert f"File: '{IMG_C_DIFFERENT_FILENAME}'" in result.stdout  # This is Entity 3, aHash dist 8
    assert "Matched by ahash" in result.stdout
    assert f"Entity ID: 3 (Matched by ahash, Distance: 8)" in result.stdout


def test_find_similar_images_dhash(test_app):
    result = runner.invoke(
        test_app, ["find-similar-images", str(_FIXTURE_IMG_A), "--dhash-threshold", "2"]
    )  # Small threshold for dhash
    if result.exit_code != 0:
        print(f"Output for dhash test:\n{result.output}")
    assert result.exit_code == 0
    assert f"Finding images similar to: {IMG_A_FILENAME}" in result.stdout
    assert "No similar images found based on the criteria." in result.stdout


def test_find_similar_images_higher_threshold_includes_more(test_app):
    # IMG_A (red) vs IMG_C_DIFFERENT (asset name for _FIXTURE_IMG_B's blue content)
    # and vs _FIXTURE_IMG_C_GREEN (green content)
    cmd = [
        "find-similar-images",
        str(_FIXTURE_IMG_A),
        "--phash-threshold",
        "60",
        "--ahash-threshold",
        "60",
        "--dhash-threshold",
        "60",
    ]
    result = runner.invoke(test_app, cmd)
    if result.exit_code != 0:
        print(f"Output for higher_threshold test:\n{result.output}")
    assert result.exit_code == 0
    # Expect _FIXTURE_IMG_A_SIMILAR, content of _FIXTURE_IMG_B (asset IMG_C_DIFFERENT_FILENAME), and _FIXTURE_IMG_C_GREEN (asset img_green_distinct.png)
    # Total 3 matches expected if thresholds are wide enough
    assert "Found 3 potentially similar image(s):" in result.stdout
    assert f"File: '{IMG_A_VERY_SIMILAR_FILENAME}'" in result.stdout
    assert f"File: '{IMG_C_DIFFERENT_FILENAME}'" in result.stdout  # This is _FIXTURE_IMG_B's content
    assert f"File: 'img_green_distinct.png'" in result.stdout  # This is _FIXTURE_IMG_C_GREEN's content


def test_find_similar_images_no_similar_found(test_app):
    # Search with _FIXTURE_IMG_B (blue), strict threshold.
    # It will be excluded from its own results.
    # _FIXTURE_IMG_A (red) and _FIXTURE_IMG_A_SIMILAR (red_with_dot) should not match blue with threshold 0.
    # _FIXTURE_IMG_C_GREEN (green) should not match blue with threshold 0.
    result = runner.invoke(
        test_app,
        [
            "find-similar-images",
            str(_FIXTURE_IMG_B),
            "--phash-threshold",
            "0",
            "--ahash-threshold",
            "0",
            "--dhash-threshold",
            "0",
        ],
    )
    if result.exit_code != 0:
        print(f"Output for no_similar_found test:\n{result.output}")
    assert result.exit_code == 0
    assert f"Finding images similar to: {IMG_B_FILENAME}" in result.stdout
    assert "No similar images found based on the criteria." in result.stdout


def test_find_similar_images_input_file_not_image(test_app):
    result = runner.invoke(test_app, ["find-similar-images", str(_FIXTURE_TXT_FILE)])
    assert result.exit_code == 1
    assert "Error processing image" in result.stdout or "Could not generate perceptual hashes" in result.stdout


def test_find_similar_images_input_file_not_exist(test_app):
    result = runner.invoke(test_app, ["find-similar-images", "non_existent_image.png"])
    assert result.exit_code == 1
    assert "Error processing image for similarity search" in result.stdout
    assert "Could not generate any perceptual hashes for non_existent_image.png" in result.stdout


# Test for add-asset to ensure MD5s are added
def test_add_asset_generates_md5(test_app):
    db_for_test = SessionLocal()
    from dam.models import Entity, ContentHashMD5Component

    # Entity for TXT_FILE is the 4th asset added in the fixture.
    # Order: IMG_A, IMG_A_SIMILAR, IMG_B (as IMG_C_DIFFERENT), TXT_FILE, IMG_C_GREEN
    # So TXT_FILE should be Entity ID 4.
    txt_entity = None
    all_props = db_for_test.query(asset_service.FilePropertiesComponent).all()
    for prop in all_props:
        if prop.original_filename == TXT_FILENAME:
            txt_entity = db_for_test.get(Entity, prop.entity_id)
            break
    assert txt_entity is not None, f"Entity for {TXT_FILENAME} not found in fixture DB."

    txt_file_md5 = get_file_md5(_FIXTURE_TXT_FILE)
    md5_components = (
        db_for_test.query(ContentHashMD5Component).filter(ContentHashMD5Component.entity_id == txt_entity.id).all()
    )
    found_md5 = any(comp.hash_value == txt_file_md5 for comp in md5_components)
    assert found_md5, f"MD5 component for {TXT_FILENAME} (Entity {txt_entity.id}) not found or mismatch."

    # Test adding a new asset
    source_files_dir = _FIXTURE_TXT_FILE.parent  # Get the temp source_files_dir
    temp_file_path = source_files_dir / "temp_add_asset_for_md5_test.txt"
    temp_file_content = "Content for MD5 test in test_add_asset_generates_md5."
    temp_file_path.write_text(temp_file_content)
    temp_file_md5_new = get_file_md5(temp_file_path)

    try:
        entity_obj, created_new = asset_service.add_asset_file(
            session=db_for_test,
            filepath_on_disk=temp_file_path,
            original_filename=temp_file_path.name,
            mime_type="text/plain",
            size_bytes=temp_file_path.stat().st_size,
        )
        db_for_test.commit()

        assert created_new, "Service should have created a new entity for a new file."

        new_md5_components = (
            db_for_test.query(ContentHashMD5Component).filter(ContentHashMD5Component.entity_id == entity_obj.id).all()
        )
        new_found_md5 = any(comp.hash_value == temp_file_md5_new for comp in new_md5_components)
        assert new_found_md5, f"MD5 component for newly added asset (Entity {entity_obj.id}) not found or mismatch."

    finally:
        temp_file_path.unlink(missing_ok=True)
        db_for_test.close()
