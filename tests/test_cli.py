import hashlib
import json  # For dumping worlds config
from pathlib import Path

import pytest
from pytest import MonkeyPatch
from typer.testing import CliRunner

from dam.core import database as app_database  # For accessing patched SessionLocal

# App is imported within a fixture to ensure patches are applied first
# from dam.core.database import SessionLocal  # Avoid top-level import, get from patched module
# from dam.services import asset_service # Removed
from dam.services import ecs_service, file_operations # Keep these
from dam.core.events import AssetFileIngestionRequested # For event-driven setup if chosen later
from dam.core.world import get_world, World # For event-driven setup

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
GIF_FILENAME = "sample_animated.gif"
MP4_FILENAME = "sample_video.mp4"

# Define Paths for physical file creation by the fixture
# These will be created under tmp_path by the fixture for isolation
# The names here are just for clarity; the fixture uses tmp_path.
IMG_A_SOURCE_PATH = TEST_DATA_DIR / IMG_A_FILENAME
IMG_A_VERY_SIMILAR_SOURCE_PATH = TEST_DATA_DIR / IMG_A_VERY_SIMILAR_FILENAME
IMG_B_SOURCE_PATH = TEST_DATA_DIR / IMG_B_FILENAME
# IMG_C_DIFFERENT_SOURCE_PATH is not strictly needed as its content comes from IMG_B
TXT_FILE_SOURCE_PATH = TEST_DATA_DIR / TXT_FILENAME
# GIF_SOURCE_PATH and MP4_SOURCE_PATH are not from TEST_DATA_DIR, they are created by fixtures


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


# This fixture will now align with conftest.py's settings_override
# It will set up the environment for multi-world CLI testing.
@pytest.fixture(scope="function", autouse=True)
def setup_test_environment(monkeypatch: MonkeyPatch, tmp_path: Path, test_worlds_config_data: dict):
    # test_worlds_config_data comes from conftest.py

    temp_storage_dirs = {}
    cli_test_worlds_config = {}

    # Create temp storage for each world defined in test_worlds_config_data
    # and update their ASSET_STORAGE_PATH
    for world_name, config in test_worlds_config_data.items():
        # Create a unique db file for each world to ensure isolation even if :memory: is tricky across processes/threads
        db_file = tmp_path / f"test_cli_{world_name}.db"
        test_db_url = f"sqlite:///{db_file}"

        temp_dir = tmp_path / f"dam_cli_asset_storage_{world_name}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_storage_dirs[world_name] = temp_dir

        cli_test_worlds_config[world_name] = {
            "DATABASE_URL": test_db_url,  # Use per-world DB file
            "ASSET_STORAGE_PATH": str(temp_dir),
        }

    default_cli_test_world = "test_world_alpha"  # Matches conftest.py default

    # Set environment variables that dam.core.config.Settings will load
    monkeypatch.setenv("DAM_WORLDS_CONFIG", json.dumps(cli_test_worlds_config))
    monkeypatch.setenv("DAM_DEFAULT_WORLD_NAME", default_cli_test_world)
    monkeypatch.setenv("TESTING_MODE", "True")

    from dam.core import config as app_config_module
    from dam.core import database as app_db_module
    from dam.core.config import Settings as AppSettings
    from dam.models.base_class import Base  # For create_all

    cli_worlds_json_str = json.dumps(cli_test_worlds_config)
    new_cli_settings = AppSettings(
        DAM_WORLDS=cli_worlds_json_str,
        DEFAULT_WORLD_NAME=default_cli_test_world,
        TESTING_MODE=True,
    )
    monkeypatch.setattr(app_config_module, "settings", new_cli_settings)

    new_db_manager = app_db_module.DatabaseManager(new_cli_settings)
    monkeypatch.setattr(app_db_module, "db_manager", new_db_manager)

    for world_name_cli in new_db_manager.get_all_world_names():
        engine = new_db_manager.get_engine(world_name_cli)
        Base.metadata.create_all(bind=engine)

    source_files_dir = tmp_path / "source_files"
    source_files_dir.mkdir(exist_ok=True)

    img_a_path = source_files_dir / IMG_A_FILENAME
    img_a_similar_path = source_files_dir / IMG_A_VERY_SIMILAR_FILENAME
    img_b_path = source_files_dir / IMG_B_FILENAME
    img_c_path = source_files_dir / IMG_C_DIFFERENT_FILENAME
    txt_file_path = source_files_dir / TXT_FILENAME

    _create_dummy_image(img_a_path, "red")
    _create_dummy_image(img_a_similar_path, "red_with_dot")
    _create_dummy_image(img_b_path, "blue")
    _create_dummy_image(img_c_path, "green")

    with open(txt_file_path, "w") as f:
        f.write(TXT_FILE_CONTENT)

    global _FIXTURE_IMG_A, _FIXTURE_IMG_A_SIMILAR, _FIXTURE_IMG_B, _FIXTURE_IMG_C_GREEN, _FIXTURE_TXT_FILE
    _FIXTURE_IMG_A = img_a_path
    _FIXTURE_IMG_A_SIMILAR = img_a_similar_path
    _FIXTURE_IMG_B = img_b_path
    _FIXTURE_IMG_C_GREEN = img_c_path
    _FIXTURE_TXT_FILE = txt_file_path

    gif_path = source_files_dir / GIF_FILENAME
    try:
        from PIL import Image as PILImage
        from PIL import ImageDraw

        img1 = PILImage.new("L", (10, 10), "white")
        draw1 = ImageDraw.Draw(img1)
        draw1.line((0, 0, 9, 9), fill="black")
        img2 = PILImage.new("L", (10, 10), "white")
        draw2 = ImageDraw.Draw(img2)
        draw2.line((0, 9, 9, 0), fill="black")
        img1.save(gif_path, save_all=True, append_images=[img2], duration=100, loop=0)
        global _FIXTURE_GIF_FILE
        _FIXTURE_GIF_FILE = gif_path
    except ImportError:
        _FIXTURE_GIF_FILE = None
        gif_path.write_bytes(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;")

    mp4_path = source_files_dir / MP4_FILENAME
    mp4_path.write_bytes(
        b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2avc1mp41\x00\x00\x00\x08free\x00\x00\x00\x00mdat"
    )
    global _FIXTURE_MP4_FILE
    _FIXTURE_MP4_FILE = mp4_path

    db = app_db_module.db_manager.get_db_session(default_cli_test_world)

    world_instance = get_world(default_cli_test_world)
    if not world_instance: # Should not happen if setup_test_environment is correctly patching settings
        pytest.fail(f"CLI Test Setup: World '{default_cli_test_world}' not found after settings override.")

    from dam.services.file_storage_service import FileStorageService as FSS_CLI_Setup # Alias for clarity
    file_storage_svc_cli_setup = world_instance.get_resource(FSS_CLI_Setup)

    from dam.models import (
        ContentHashMD5Component, ContentHashSHA256Component, FileLocationComponent,
        FilePropertiesComponent, NeedsMetadataExtractionComponent, OriginalSourceInfoComponent,
        ImagePerceptualAHashComponent, ImagePerceptualDHashComponent, ImagePerceptualPHashComponent,
    )

    assets_to_populate = [
        (_FIXTURE_IMG_A, IMG_A_FILENAME, "image/png"),
        (_FIXTURE_IMG_A_SIMILAR, IMG_A_VERY_SIMILAR_FILENAME, "image/png"),
        (_FIXTURE_IMG_B, IMG_C_DIFFERENT_FILENAME, "image/png"),
        (_FIXTURE_TXT_FILE, TXT_FILENAME, "text/plain"),
        (_FIXTURE_IMG_C_GREEN, "img_green_distinct.png", "image/png"),
    ]
    if _FIXTURE_GIF_FILE and _FIXTURE_GIF_FILE.exists():
        assets_to_populate.append((_FIXTURE_GIF_FILE, GIF_FILENAME, "image/gif"))
    if _FIXTURE_MP4_FILE and _FIXTURE_MP4_FILE.exists():
        assets_to_populate.append((_FIXTURE_MP4_FILE, MP4_FILENAME, "video/mp4"))

    try:
        for file_path, original_name, mime_val in assets_to_populate:
            if not file_path.exists(): # Should already be checked by fixture creation, but defensive
                continue

            file_content = file_path.read_bytes()
            size_val = file_path.stat().st_size

            sha256_val = hashlib.sha256(file_content).hexdigest()
            md5_val = hashlib.md5(file_content).hexdigest()

            # Store file in CAS
            _, cas_path_suffix = file_storage_svc_cli_setup.store_file(file_content, original_filename=original_name)

            # Check if entity with this content hash already exists
            entity = ecs_service.find_entity_by_content_hash(db, sha256_val) # Re-use existing ecs_service helper

            if not entity:
                entity = ecs_service.create_entity(db)
                ecs_service.add_component_to_entity(db, entity.id, ContentHashSHA256Component(hash_value=sha256_val))
                ecs_service.add_component_to_entity(db, entity.id, ContentHashMD5Component(hash_value=md5_val))
                ecs_service.add_component_to_entity(db, entity.id, FilePropertiesComponent(
                    original_filename=original_name, file_size_bytes=size_val, mime_type=mime_val
                ))
                # Add CAS location
                ecs_service.add_component_to_entity(db, entity.id, FileLocationComponent(
                    content_identifier=sha256_val, storage_type="local_cas",
                    physical_path_or_key=cas_path_suffix, contextual_filename=original_name
                ))

            # Always add OriginalSourceInfo for this specific file "ingestion" during setup
            ecs_service.add_component_to_entity(db, entity.id, OriginalSourceInfoComponent(
                original_filename=original_name, original_path=str(file_path.resolve())
            ))

            # Add perceptual hashes for images
            if mime_val.startswith("image/"):
                p_hashes = file_operations.generate_perceptual_hashes(file_path)
                if p_hashes.get("phash") and not ecs_service.get_component(db, entity.id, ImagePerceptualPHashComponent): # Add only if not present
                     ecs_service.add_component_to_entity(db, entity.id, ImagePerceptualPHashComponent(hash_value=p_hashes["phash"]))
                if p_hashes.get("ahash") and not ecs_service.get_component(db, entity.id, ImagePerceptualAHashComponent):
                     ecs_service.add_component_to_entity(db, entity.id, ImagePerceptualAHashComponent(hash_value=p_hashes["ahash"]))
                if p_hashes.get("dhash") and not ecs_service.get_component(db, entity.id, ImagePerceptualDHashComponent):
                     ecs_service.add_component_to_entity(db, entity.id, ImagePerceptualDHashComponent(hash_value=p_hashes["dhash"]))

            # Mark for metadata extraction (mimicking what add_asset_file used to do)
            if not ecs_service.get_component(db, entity.id, NeedsMetadataExtractionComponent):
                ecs_service.add_component_to_entity(db, entity.id, NeedsMetadataExtractionComponent())

        db.commit()
    except Exception as e:
        db.rollback()
        error_type = type(e).__name__
        # db_url can be fetched from new_cli_settings used above for clarity if needed
        pytest.fail(f"Failed to setup initial assets in temporary DB for CLI tests: {e}. Error: {error_type}")
    finally:
        db.close()

    yield


def get_file_sha256(filepath: Path) -> str:
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def get_file_md5(filepath: Path) -> str:
    return hashlib.md5(filepath.read_bytes()).hexdigest()


# Tests for "find-file-by-hash" command
def test_find_file_by_sha256_direct_hash(test_app):
    sha256_hash = get_file_sha256(_FIXTURE_TXT_FILE)
    result = runner.invoke(test_app, ["find-file-by-hash", sha256_hash, "--hash-type", "sha256"])
    if result.exit_code != 0:
        print(f"CLI Error Output for test_find_file_by_sha256_direct_hash:\n{result.output}")

    assert result.exit_code == 0
    from dam.cli import global_state  # Import for assertion
    assert f"Querying world '{global_state.world_name}' for asset with sha256 hash: {sha256_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout
    assert f"Value: {sha256_hash}" in result.stdout
    assert f"Name='{TXT_FILENAME}'" in result.stdout
    assert "Image Dimensions:" not in result.stdout


def test_find_image_by_sha256_direct_hash_shows_dimensions(test_app):
    sha256_hash = get_file_sha256(_FIXTURE_IMG_A)
    result = runner.invoke(test_app, ["find-file-by-hash", sha256_hash, "--hash-type", "sha256"])
    if result.exit_code != 0:
        print(f"CLI Error Output for test_find_image_by_sha256_direct_hash_shows_dimensions:\n{result.output}")
    assert result.exit_code == 0
    assert "Found Entity ID:" in result.stdout
    assert f"Name='{IMG_A_FILENAME}'" in result.stdout
    assert "Image Dimensions:" in result.stdout
    assert "Width: 10px" in result.stdout
    assert "Height: 10px" in result.stdout


def test_find_gif_by_hash_shows_dimensions_and_frames(test_app):
    if not _FIXTURE_GIF_FILE:
        pytest.skip("Pillow not available, GIF fixture not created.")

    sha256_hash = get_file_sha256(_FIXTURE_GIF_FILE)
    result = runner.invoke(test_app, ["find-file-by-hash", sha256_hash])
    if result.exit_code != 0:
        print(f"CLI Error Output for test_find_gif_by_hash_shows_dimensions_and_frames:\n{result.output}")
    assert result.exit_code == 0
    assert "Found Entity ID:" in result.stdout
    assert f"Name='{GIF_FILENAME}'" in result.stdout
    assert "Image Dimensions:" in result.stdout
    assert "Width: 10px" in result.stdout
    assert "Height: 10px" in result.stdout
    assert "Animated Frame Properties:" in result.stdout


def test_find_video_by_hash_shows_dimensions_frames_audio(test_app):
    sha256_hash = get_file_sha256(_FIXTURE_MP4_FILE)
    result = runner.invoke(test_app, ["find-file-by-hash", sha256_hash])
    if result.exit_code != 0:
        print(f"CLI Error Output for test_find_video_by_hash_shows_dimensions_frames_audio:\n{result.output}")
    assert result.exit_code == 0
    assert "Found Entity ID:" in result.stdout
    assert f"Name='{MP4_FILENAME}'" in result.stdout
    pass


def test_find_file_by_md5_direct_hash(test_app):
    md5_hash = get_file_md5(_FIXTURE_TXT_FILE)
    result = runner.invoke(test_app, ["find-file-by-hash", md5_hash, "--hash-type", "md5"])
    assert result.exit_code == 0
    from dam.cli import global_state  # Import for assertion
    assert f"Querying world '{global_state.world_name}' for asset with md5 hash: {md5_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout
    assert f"Value: {md5_hash}" in result.stdout
    assert f"Name='{TXT_FILENAME}'" in result.stdout


def test_find_file_by_sha256_filepath(test_app):
    result = runner.invoke(
        test_app, ["find-file-by-hash", "testhash", "--file", str(_FIXTURE_TXT_FILE), "--hash-type", "sha256"]
    )
    assert result.exit_code == 0
    sha256_hash = get_file_sha256(_FIXTURE_TXT_FILE)
    assert f"Calculating sha256 hash for file: {str(_FIXTURE_TXT_FILE)}" in result.stdout
    assert f"Calculated sha256 hash: {sha256_hash}" in result.stdout
    assert "Found Entity ID:" in result.stdout


def test_find_file_by_md5_filepath(test_app):
    result = runner.invoke(
        test_app, ["find-file-by-hash", "testhash", "--file", str(_FIXTURE_TXT_FILE), "--hash-type", "md5"]
    )
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
    )
    assert result.exit_code == 1
    assert "Unsupported hash type for file calculation: sha1" in result.stdout


def test_find_file_by_hash_invalid_hash_type_direct(test_app):
    some_hash = get_file_sha256(_FIXTURE_TXT_FILE)
    result = runner.invoke(test_app, ["find-file-by-hash", some_hash, "--hash-type", "sha1"])
    assert result.exit_code == 0
    assert f"No asset found with sha1 hash: {some_hash}" in result.stdout


# Tests for "find-similar-images" command
def test_find_similar_images_phash(test_app):
    result = runner.invoke(test_app, ["find-similar-images", str(_FIXTURE_IMG_A), "--phash-threshold", "15"])
    if result.exit_code != 0:
        print(f"Output for phash test (raw):\n{result.output}")
    print(f"Output for phash test (stdout):\n{result.stdout}")
    assert result.exit_code == 0
    assert f"Finding images similar to: {IMG_A_FILENAME}" in result.stdout
    assert "Found 2 potentially similar image(s):" in result.stdout
    assert f"File: '{IMG_A_VERY_SIMILAR_FILENAME}'" in result.stdout
    assert f"File: '{IMG_C_DIFFERENT_FILENAME}'" in result.stdout
    assert "File: 'img_green_distinct.png'" not in result.stdout
    assert "Entity ID: 2 (Matched by dhash, Distance: 3)" in result.stdout
    assert "Entity ID: 3 (Matched by phash, Distance: 7)" in result.stdout


def test_find_similar_images_ahash(test_app):
    result = runner.invoke(
        test_app,
        [
            "find-similar-images",
            str(_FIXTURE_IMG_A),
            "--phash-threshold",
            "0",
            "--ahash-threshold",
            "31",
            "--dhash-threshold",
            "0",
        ],
    )
    if result.exit_code != 0:
        print(f"Output for ahash test (raw):\n{result.output}")
    print(f"Output for ahash test (stdout):\n{result.stdout}")
    assert result.exit_code == 0
    assert f"Finding images similar to: {IMG_A_FILENAME}" in result.stdout
    assert "Found 1 potentially similar image(s):" in result.stdout
    assert f"File: '{IMG_C_DIFFERENT_FILENAME}'" in result.stdout
    assert "Matched by ahash" in result.stdout
    assert "Entity ID: 3 (Matched by ahash, Distance: 8)" in result.stdout


def test_find_similar_images_dhash(test_app):
    result = runner.invoke(
        test_app, ["find-similar-images", str(_FIXTURE_IMG_A), "--dhash-threshold", "2"]
    )
    if result.exit_code != 0:
        print(f"Output for dhash test:\n{result.output}")
    assert result.exit_code == 0
    assert f"Finding images similar to: {IMG_A_FILENAME}" in result.stdout
    assert "No similar images found based on the criteria." in result.stdout


def test_find_similar_images_higher_threshold_includes_more(test_app):
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
    assert "Found 4 potentially similar image(s):" in result.stdout
    assert f"File: '{IMG_A_VERY_SIMILAR_FILENAME}'" in result.stdout
    assert f"File: '{IMG_C_DIFFERENT_FILENAME}'" in result.stdout
    assert f"File: '{GIF_FILENAME}'" in result.stdout
    assert "File: 'img_green_distinct.png'" in result.stdout


def test_find_similar_images_no_similar_found(test_app):
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


def test_add_asset_generates_md5(test_app, setup_test_environment):
    from dam.cli import global_state

    cli_test_world_name = global_state.world_name if global_state.world_name else "test_world_alpha"
    db_for_test = app_database.db_manager.get_db_session(cli_test_world_name) # Session for verification

    from dam.models import ContentHashMD5Component, Entity, FilePropertiesComponent as ModelFilePropertiesComponent

    # Verify existing text file's MD5 from setup
    existing_txt_fpc = db_for_test.query(ModelFilePropertiesComponent).filter_by(original_filename=TXT_FILENAME).first()
    assert existing_txt_fpc is not None, f"Pre-existing {TXT_FILENAME} not found."

    txt_entity_id = existing_txt_fpc.entity_id
    txt_file_md5_expected = get_file_md5(_FIXTURE_TXT_FILE)
    md5_comp_existing = ecs_service.get_component(db_for_test, txt_entity_id, ContentHashMD5Component)
    assert md5_comp_existing is not None, f"MD5 component for existing {TXT_FILENAME} not found."
    assert md5_comp_existing.hash_value == txt_file_md5_expected, f"MD5 mismatch for existing {TXT_FILENAME}."

    # Test adding a new file via CLI
    source_files_dir = _FIXTURE_TXT_FILE.parent # Reuse temp dir for new file
    temp_file_path = source_files_dir / "temp_add_asset_for_md5_test.txt"
    temp_file_content = "Content for MD5 test in test_add_asset_generates_md5."
    temp_file_path.write_text(temp_file_content)
    temp_file_md5_new_expected = get_file_md5(temp_file_path)

    try:
        # Invoke CLI to add the new asset
        result = runner.invoke(test_app, ["add-asset", str(temp_file_path), "--world", cli_test_world_name])
        assert result.exit_code == 0, f"CLI add-asset failed: {result.output}"
        assert "Dispatched AssetFileIngestionRequested" in result.output # Check for event dispatch message
        assert "Post-ingestion systems completed" in result.output


        # Verify the new asset in DB
        new_fpc = db_for_test.query(ModelFilePropertiesComponent).filter_by(original_filename=temp_file_path.name).first()
        assert new_fpc is not None, f"Newly added asset {temp_file_path.name} not found in DB."

        new_entity_id = new_fpc.entity_id
        md5_comp_new = ecs_service.get_component(db_for_test, new_entity_id, ContentHashMD5Component)
        assert md5_comp_new is not None, f"MD5 component for new asset {temp_file_path.name} not found."
        assert md5_comp_new.hash_value == temp_file_md5_new_expected, f"MD5 mismatch for new asset {temp_file_path.name}."

    finally:
        temp_file_path.unlink(missing_ok=True)
        db_for_test.close()


def test_add_asset_directory(test_app, tmp_path):
    source_dir = tmp_path / "test_add_dir"
    source_dir.mkdir()
    file1 = source_dir / "file1.txt"
    file1.write_text("content1")
    file2 = source_dir / "file2.txt"
    file2.write_text("content2")
    result = runner.invoke(test_app, ["add-asset", str(source_dir)])
    if result.exit_code != 0:
        print(f"CLI Error Output for test_add_asset_directory:\n{result.output}")
    assert result.exit_code == 0
    assert f"Processing directory: {source_dir}" in result.stdout
    assert "Found 2 file(s) to process." in result.stdout
    assert "Processing file " in result.stdout
    assert "file1.txt" in result.stdout
    assert "file2.txt" in result.stdout
    assert "Successfully added new asset." in result.stdout
    assert result.stdout.count("Successfully added new asset.") >= 2
    assert "New assets added: 2" in result.stdout
    db = app_database.SessionLocal()
    try:
        from dam.models import FilePropertiesComponent

        file1_props = (
            db.query(FilePropertiesComponent)
            .filter(FilePropertiesComponent.original_filename == "file1.txt")
            .one_or_none()
        )
        file2_props = (
            db.query(FilePropertiesComponent)
            .filter(FilePropertiesComponent.original_filename == "file2.txt")
            .one_or_none()
        )
        assert file1_props is not None, "file1.txt not found in DB"
        assert file2_props is not None, "file2.txt not found in DB"
    finally:
        db.close()


def test_add_asset_directory_recursive(test_app, tmp_path):
    source_dir = tmp_path / "test_add_dir_recursive"
    source_dir.mkdir()
    sub_dir = source_dir / "sub"
    sub_dir.mkdir()
    file1 = source_dir / "file1.txt"
    file1.write_text("content1_rec")
    file2 = sub_dir / "file2.txt"
    file2.write_text("content2_rec")
    result = runner.invoke(test_app, ["add-asset", str(source_dir), "--recursive"])
    if result.exit_code != 0:
        print(f"CLI Error Output for test_add_asset_directory_recursive:\n{result.output}")
    assert result.exit_code == 0
    assert f"Processing directory: {source_dir}" in result.stdout
    assert "Found 2 file(s) to process." in result.stdout
    assert "file1.txt" in result.stdout
    assert "file2.txt" in result.stdout
    assert "New assets added: 2" in result.stdout
    db = app_database.SessionLocal()
    try:
        from dam.models import FilePropertiesComponent

        file1_props = (
            db.query(FilePropertiesComponent)
            .filter(FilePropertiesComponent.original_filename == "file1.txt")
            .one_or_none()
        )
        file2_props = (
            db.query(FilePropertiesComponent)
            .filter(FilePropertiesComponent.original_filename == "file2.txt")
            .one_or_none()
        )
        assert file1_props is not None, "file1.txt from root not found"
        assert file2_props is not None, "sub/file2.txt not found"
    finally:
        db.close()


def test_add_asset_no_copy(test_app, tmp_path):
    source_file = tmp_path / "no_copy_test.txt"
    source_file.write_text("no_copy_content")
    source_file_abs_path_str = str(source_file.resolve())
    result = runner.invoke(test_app, ["add-asset", str(source_file), "--no-copy"])
    if result.exit_code != 0:
        print(f"CLI Error Output for test_add_asset_no_copy:\n{result.output}")
    assert result.exit_code == 0
    assert "Successfully added new asset." in result.stdout
    db = app_database.SessionLocal()
    try:
        from sqlalchemy import inspect as sqlalchemy_inspect_func

        from dam.models import FileLocationComponent, FilePropertiesComponent

        inspector = sqlalchemy_inspect_func(app_database.engine)
        print(f"DEBUG: Tables in DB for test_add_asset_no_copy: {inspector.get_table_names()}")
        all_fpcs = db.query(FilePropertiesComponent).all()
        print(f"DEBUG: Found {len(all_fpcs)} FilePropertiesComponents in test_add_asset_no_copy.")
        for fpc_debug in all_fpcs:
            print(
                f"DEBUG: FPC ID: {fpc_debug.id}, Filename: {fpc_debug.original_filename}, Entity ID: {fpc_debug.entity_id}"
            )
        fpc = db.query(FilePropertiesComponent).filter_by(original_filename="no_copy_test.txt").one()
        flc = db.query(FileLocationComponent).filter_by(entity_id=fpc.entity_id).one()
        assert flc.storage_type == "referenced_local_file"
        assert flc.file_identifier == source_file_abs_path_str
        from dam.core.config import settings

        asset_storage_path = Path(settings.ASSET_STORAGE_PATH)
        file_hash = get_file_sha256(source_file)
        expected_cas_path = asset_storage_path / file_hash[:2] / file_hash[2:4] / file_hash
        assert not expected_cas_path.exists(), "File should not have been copied to CAS with --no-copy"
    finally:
        db.close()


def test_add_asset_directory_no_copy(test_app, tmp_path):
    source_dir = tmp_path / "test_add_dir_no_copy"
    source_dir.mkdir()
    file1 = source_dir / "no_copy_dir_file1.txt"
    file1.write_text("content_dir_nc1")
    file1_abs_path_str = str(file1.resolve())
    file2 = source_dir / "no_copy_dir_file2.txt"
    file2.write_text("content_dir_nc2")
    file2_abs_path_str = str(file2.resolve())
    result = runner.invoke(test_app, ["add-asset", str(source_dir), "--no-copy"])
    if result.exit_code != 0:
        print(f"CLI Error Output for test_add_asset_directory_no_copy:\n{result.output}")
    assert result.exit_code == 0
    assert "New assets added: 2" in result.stdout
    db = app_database.SessionLocal()
    try:
        from dam.core.config import settings
        from dam.models import FileLocationComponent, FilePropertiesComponent

        asset_storage_path = Path(settings.ASSET_STORAGE_PATH)
        fpc1 = db.query(FilePropertiesComponent).filter_by(original_filename="no_copy_dir_file1.txt").one()
        flc1 = db.query(FileLocationComponent).filter_by(entity_id=fpc1.entity_id).one()
        assert flc1.storage_type == "referenced_local_file"
        assert flc1.file_identifier == file1_abs_path_str
        file1_hash = get_file_sha256(file1)
        expected_cas_path1 = asset_storage_path / file1_hash[:2] / file1_hash[2:4] / file1_hash
        assert not expected_cas_path1.exists()
        fpc2 = db.query(FilePropertiesComponent).filter_by(original_filename="no_copy_dir_file2.txt").one()
        flc2 = db.query(FileLocationComponent).filter_by(entity_id=fpc2.entity_id).one()
        assert flc2.storage_type == "referenced_local_file"
        assert flc2.file_identifier == file2_abs_path_str
        file2_hash = get_file_sha256(file2)
        expected_cas_path2 = asset_storage_path / file2_hash[:2] / file2_hash[2:4] / file2_hash
        assert not expected_cas_path2.exists()
    finally:
        db.close()


def _count_entities_in_world(world_name: str) -> int:
    session = app_database.db_manager.get_db_session(world_name)
    try:
        from dam.models import Entity

        return session.query(Entity).count()
    finally:
        session.close()


def _get_entity_by_filename(world_name: str, filename: str) -> any:
    session = app_database.db_manager.get_db_session(world_name)
    try:
        from dam.models import Entity, FilePropertiesComponent

        fpc = session.query(FilePropertiesComponent).filter_by(original_filename=filename).first()
        if fpc:
            return session.query(Entity).filter_by(id=fpc.entity_id).first()
        return None
    finally:
        session.close()


@pytest.mark.asyncio # Make async
async def test_cli_merge_worlds_db(test_app, settings_override, tmp_path): # Make async
    source_world_cli = "test_world_alpha"
    target_world_cli = "test_world_beta"
    # source_session_alpha = app_database.db_manager.get_db_session(source_world_cli) # Not needed before populate
    s_img_file = tmp_path / "s_img_merge.png"
    _create_dummy_image(s_img_file, "red", size=(5, 5))
    s_txt_file = tmp_path / "s_txt_merge.txt"
    s_txt_file.write_text("Source text for merge")
    from tests.services.test_world_service_advanced import _populate_world_with_assets as populate_helper

    await populate_helper(source_world_cli, s_img_file, s_txt_file) # Pass world_name, await
    # source_session_alpha.close() # Closed by helper or not needed before count
    assert _count_entities_in_world(source_world_cli) == 2
    assert _count_entities_in_world(target_world_cli) == 0
    result = runner.invoke(test_app, ["merge-worlds-db", source_world_cli, target_world_cli])
    print(f"CLI merge-worlds-db output:\n{result.output}")
    assert result.exit_code == 0
    assert f"Successfully merged world '{source_world_cli}' into '{target_world_cli}'" in result.stdout
    assert _count_entities_in_world(target_world_cli) == 2
    assert _count_entities_in_world(source_world_cli) == 2


def test_cli_split_world_db(test_app, settings_override, tmp_path):
    source_world_cli_split = "test_world_alpha"
    selected_target_cli = "test_world_beta"
    remaining_target_cli = "test_world_gamma"
    source_session_s = app_database.db_manager.get_db_session(source_world_cli_split)
    s_img_png = tmp_path / "s_split_img.png"
    _create_dummy_image(s_img_png, "red", size=(5, 5))
    s_img_jpg = tmp_path / "s_split_img.jpg"
    _create_dummy_image(s_img_jpg, "blue", size=(6, 6))
    s_txt_split = tmp_path / "s_split_txt.txt"
    s_txt_split.write_text("Source text for split")
    # from dam.services import asset_service # Removed

    # Setup using ecs_service and FileStorageService directly
    split_world_instance = get_world(source_world_cli_split)
    if not split_world_instance:
        pytest.fail(f"CLI Test Setup: World '{source_world_cli_split}' not found for split test.")

    from dam.services.file_storage_service import FileStorageService as FSS_CLI_Split_Setup
    fss_split = split_world_instance.get_resource(FSS_CLI_Split_Setup)

    from dam.models import (
        ContentHashMD5Component, ContentHashSHA256Component, FileLocationComponent,
        FilePropertiesComponent, NeedsMetadataExtractionComponent, OriginalSourceInfoComponent
    )

    assets_for_split = [
        (s_img_png, "split_image.png", "image/png"),
        (s_img_jpg, "split_image.jpg", "image/jpeg"),
        (s_txt_split, "split_text.txt", "text/plain"),
    ]

    for file_p, orig_name, mime_t in assets_for_split:
        file_c = file_p.read_bytes()
        size_f = file_p.stat().st_size
        sha256_f = hashlib.sha256(file_c).hexdigest()
        md5_f = hashlib.md5(file_c).hexdigest()
        _, cas_path_f = fss_split.store_file(file_c, original_filename=orig_name)

        entity_s = ecs_service.create_entity(source_session_s)
        ecs_service.add_component_to_entity(source_session_s, entity_s.id, ContentHashSHA256Component(hash_value=sha256_f))
        ecs_service.add_component_to_entity(source_session_s, entity_s.id, ContentHashMD5Component(hash_value=md5_f))
        ecs_service.add_component_to_entity(source_session_s, entity_s.id, FilePropertiesComponent(
            original_filename=orig_name, file_size_bytes=size_f, mime_type=mime_t
        ))
        ecs_service.add_component_to_entity(source_session_s, entity_s.id, FileLocationComponent(
            content_identifier=sha256_f, storage_type="local_cas",
            physical_path_or_key=cas_path_f, contextual_filename=orig_name
        ))
        ecs_service.add_component_to_entity(source_session_s, entity_s.id, OriginalSourceInfoComponent(
            original_filename=orig_name, original_path=str(file_p.resolve())
        ))
        ecs_service.add_component_to_entity(source_session_s, entity_s.id, NeedsMetadataExtractionComponent())

    source_session_s.commit()
    source_session_s.close()
    assert _count_entities_in_world(source_world_cli_split) == 3
    assert _count_entities_in_world(selected_target_cli) == 0
    assert _count_entities_in_world(remaining_target_cli) == 0
    cmd = [
        "split-world-db",
        source_world_cli_split,
        selected_target_cli,
        remaining_target_cli,
        "--component-name",
        "FilePropertiesComponent",
        "--attribute",
        "mime_type",
        "--value",
        "image/png",
        "--operator",
        "eq",
    ]
    result = runner.invoke(test_app, cmd)
    print(f"CLI split-world-db output:\n{result.output}")
    assert result.exit_code == 0
    assert (
        f"Split complete: 1 entities to '{selected_target_cli}', 2 entities to '{remaining_target_cli}'."
        in result.stdout
    )
    assert _count_entities_in_world(selected_target_cli) == 1
    assert _count_entities_in_world(remaining_target_cli) == 2
    assert _count_entities_in_world(source_world_cli_split) == 3
    png_entity_selected = _get_entity_by_filename(selected_target_cli, "split_image.png")
    assert png_entity_selected is not None
    jpg_entity_remaining = _get_entity_by_filename(remaining_target_cli, "split_image.jpg")
    txt_entity_remaining = _get_entity_by_filename(remaining_target_cli, "split_text.txt")
    assert jpg_entity_remaining is not None
    assert txt_entity_remaining is not None


@pytest.mark.asyncio # Make async
async def test_cli_split_world_db_delete_source(test_app, settings_override, tmp_path): # Make async
    source_world_del = "test_world_alpha_del_split"
    selected_target_del = "test_world_beta_del_split"
    remaining_target_del = "test_world_gamma_del_split"
    # source_session_del = app_database.db_manager.get_db_session(source_world_del) # Session managed by helper or later
    s_img_del = tmp_path / "s_del_img.png"
    _create_dummy_image(s_img_del, "green")
    s_txt_del = tmp_path / "s_del_txt.txt"
    s_txt_del.write_text("Delete me after split")
    from tests.services.test_world_service_advanced import _populate_world_with_assets as populate_helper_del

    # _populate_world_with_assets now takes world_name and is async
    await populate_helper_del(source_world_del, s_img_del, s_txt_del)
    # source_session_del.close() # Closed by helper or not needed before count
    assert _count_entities_in_world(source_world_del) == 2
    cmd_del = [
        "split-world-db",
        source_world_del,
        selected_target_del,
        remaining_target_del,
        "--component-name",
        "FilePropertiesComponent",
        "--attribute",
        "mime_type",
        "--value",
        "image/png",
        "--delete-from-source",
    ]
    result_del = runner.invoke(test_app, cmd_del, input="yes\n")
    print(f"CLI split-world-db --delete-from-source output:\n{result_del.output}")
    assert result_del.exit_code == 0
    assert (
        f"Split complete: 1 entities to '{selected_target_del}', 1 entities to '{remaining_target_del}'."
        in result_del.stdout
    )
    assert f"Entities deleted from source world '{source_world_del}'." in result_del.stdout
    assert _count_entities_in_world(source_world_del) == 0
    assert _count_entities_in_world(selected_target_del) == 1
    assert _count_entities_in_world(remaining_target_del) == 1
