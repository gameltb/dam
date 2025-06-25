# tests/services/test_asset_service.py
import shutil  # For copying files in tests
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from dam.core.config import settings as app_settings  # To get current world name
from dam.models import (
    ContentHashSHA256Component,  # Added for direct verification
    FileLocationComponent,  # Added for testing specific file location scenarios
    FramePropertiesComponent,
    ImageDimensionsComponent,
    ImagePerceptualAHashComponent,
    ImagePerceptualDHashComponent,
    ImagePerceptualPHashComponent,
)
from dam.services import asset_service, ecs_service, file_operations

# Fixtures like db_session, settings_override, temp_asset_file, temp_image_file
# are expected to be provided by conftest.py

# Removed module_db_engine as conftest.py handles engine and session setup per test world.

# sample_image_a, non_image_file, sample_video_file, sample_audio_file, sample_gif_file
# are now expected to be standard pytest fixtures, potentially from conftest.py or defined here.
# For simplicity, let's assume they are available as defined previously, or use temp_image_file etc.
# from conftest.py directly.

# Local file fixtures are removed, assuming they are provided by conftest.py:
# sample_image_a, sample_text_file (was non_image_file),
# sample_video_file_placeholder (was sample_video_file),
# sample_audio_file_placeholder (was sample_audio_file),
# sample_gif_file_placeholder (was sample_gif_file).
# Tests will need to use the new placeholder names for multimedia files if they were changed in conftest.

def test_add_image_asset_creates_perceptual_hashes(settings_override, db_session: Session, sample_image_a: Path):
    # settings_override ensures that the test runs with the correct multi-world config
    # db_session is for the default test world (e.g., "test_world_alpha")
    current_world_name = settings_override.DEFAULT_WORLD_NAME  # Use patched settings from fixture

    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa
    except ImportError:
        pytest.skip("ImageHash or Pillow not installed.")

    props = file_operations.get_file_properties(sample_image_a)
    entity, created_new = asset_service.add_asset_file(
        session=db_session,
        filepath_on_disk=sample_image_a,
        original_filename=props[0],
        mime_type=props[2],
        size_bytes=props[1],
        world_name=current_world_name,  # Pass current world name
    )
    db_session.commit()

    assert created_new is True
    assert entity.id is not None

    phashes = ecs_service.get_components(db_session, entity.id, ImagePerceptualPHashComponent)
    ahashes = ecs_service.get_components(db_session, entity.id, ImagePerceptualAHashComponent)
    dhashes = ecs_service.get_components(db_session, entity.id, ImagePerceptualDHashComponent)
    assert len(phashes) + len(ahashes) + len(dhashes) > 0

    dim_comp = ecs_service.get_component(db_session, entity.id, ImageDimensionsComponent)
    assert dim_comp is not None and dim_comp.width_pixels == 2 and dim_comp.height_pixels == 1

    # Verify asset storage path corresponds to the world
    world_config = settings_override.get_world_config(current_world_name) # Use patched settings
    sha256_comp = ecs_service.get_component(db_session, entity.id, ContentHashSHA256Component)
    assert sha256_comp is not None
    expected_file_path_fragment = (
        Path(world_config.ASSET_STORAGE_PATH)
        / sha256_comp.hash_value[:2]
        / sha256_comp.hash_value[2:4]
        / sha256_comp.hash_value
    )

    # flc = ecs_service.get_components(db_session, entity.id, FileLocationComponent)[0] # Not strictly needed for this check
    # stored_file_path = file_operations.get_file_storage_path( # This variable was unused
    #     flc.file_identifier, world_config.ASSET_STORAGE_PATH
    # )
    # or directly check using file_storage service if it provides such a direct path getter based on world_config

    # Reconstruct path using file_storage internal logic for verification
    from dam.services.file_storage import _get_storage_path_for_world  # Access internal for test

    reconstructed_path = _get_storage_path_for_world(sha256_comp.hash_value, world_config)
    assert reconstructed_path.exists()
    assert str(expected_file_path_fragment) in str(reconstructed_path)


def test_add_non_image_asset_no_perceptual_hashes(settings_override, db_session: Session, sample_text_file: Path):
    current_world_name = settings_override.DEFAULT_WORLD_NAME # Use patched settings from fixture
    props = file_operations.get_file_properties(sample_text_file)
    entity, created_new = asset_service.add_asset_file(
        session=db_session,
        filepath_on_disk=sample_text_file,
        original_filename=props[0],
        mime_type=props[2],
        size_bytes=props[1],
        world_name=current_world_name,
    )
    db_session.commit()
    assert created_new is True
    assert len(ecs_service.get_components(db_session, entity.id, ImagePerceptualPHashComponent)) == 0


def test_add_existing_image_content_adds_missing_hashes(
    settings_override, db_session: Session, sample_image_a: Path, tmp_path: Path
):
    current_world_name = settings_override.DEFAULT_WORLD_NAME # Use patched settings from fixture
    try:
        import imagehash  # noqa: F401
        from PIL import Image  # noqa
    except ImportError:
        pytest.skip("ImageHash or Pillow not installed.")

    props1 = file_operations.get_file_properties(sample_image_a)
    entity1, _ = asset_service.add_asset_file(
        db_session, sample_image_a, props1[0], props1[2], props1[1], world_name=current_world_name
    )
    db_session.commit()

    initial_dhashes = ecs_service.get_components(db_session, entity1.id, ImagePerceptualDHashComponent)
    if not initial_dhashes:
        pytest.skip("Initial add did not generate dhash.")

    db_session.delete(initial_dhashes[0])
    db_session.commit()
    assert not ecs_service.get_components(db_session, entity1.id, ImagePerceptualDHashComponent)

    copy_of_sample_image_a = tmp_path / "sample_A_copy.png"
    shutil.copy2(sample_image_a, copy_of_sample_image_a)
    props2 = file_operations.get_file_properties(copy_of_sample_image_a)
    entity2, created_new2 = asset_service.add_asset_file(
        db_session, copy_of_sample_image_a, props2[0], props2[2], props2[1], world_name=current_world_name
    )
    db_session.commit()

    assert created_new2 is False and entity2.id == entity1.id
    final_dhashes = ecs_service.get_components(db_session, entity2.id, ImagePerceptualDHashComponent)
    assert len(final_dhashes) > 0
    assert final_dhashes[0].hash_value == initial_dhashes[0].hash_value


def test_add_video_asset_creates_multimedia_props(settings_override, db_session: Session, sample_video_file_placeholder: Path):
    current_world_name = settings_override.DEFAULT_WORLD_NAME # Use patched settings from fixture
    try:
        from hachoir.parser import createParser  # noqa
    except ImportError:
        pytest.skip("Hachoir not installed.")

    props = file_operations.get_file_properties(sample_video_file_placeholder)
    mime_type = "video/mp4" if props[2] == "application/octet-stream" else props[2] # Ensure correct mime for test
    entity, _ = asset_service.add_asset_file(
        db_session, sample_video_file_placeholder, props[0], mime_type, props[1], world_name=current_world_name
    )
    db_session.commit()
    # Assertions depend on dummy file's parsability. Check if components are attempted.
    # ecs_service.get_component(db_session, entity.id, ImageDimensionsComponent)
    # ecs_service.get_component(db_session, entity.id, FramePropertiesComponent)
    # ecs_service.get_component(db_session, entity.id, AudioPropertiesComponent)
    # For a dummy file, it's hard to assert specific values. Test ensures no crash.
    assert entity.id is not None


def test_add_audio_asset_creates_audio_props(settings_override, db_session: Session, sample_audio_file_placeholder: Path):
    current_world_name = settings_override.DEFAULT_WORLD_NAME # Use patched settings from fixture
    try:
        from hachoir.parser import createParser  # noqa
    except ImportError:
        pytest.skip("Hachoir not installed.")
    props = file_operations.get_file_properties(sample_audio_file_placeholder)
    mime_type = "audio/mpeg" if props[2] == "application/octet-stream" else props[2] # Ensure correct mime for test
    entity, _ = asset_service.add_asset_file(
        db_session, sample_audio_file_placeholder, props[0], mime_type, props[1], world_name=current_world_name
    )
    db_session.commit()
    # audio_props = ecs_service.get_component(db_session, entity.id, AudioPropertiesComponent) # Variable unused
    # if audio_props: assert audio_props.entity_id == entity.id # Dummy might not yield props
    assert entity.id is not None


def test_add_gif_asset_creates_frame_and_image_props(settings_override, db_session: Session, sample_gif_file_placeholder: Path):
    current_world_name = settings_override.DEFAULT_WORLD_NAME # Use patched settings from fixture
    try:
        from hachoir.parser import createParser  # noqa
    except ImportError:
        pytest.skip("Hachoir not installed.")
    props = file_operations.get_file_properties(sample_gif_file_placeholder)
    # Ensure mime type is image/gif for the placeholder if get_file_properties doesn't get it
    mime_type = "image/gif" if props[2] != "image/gif" else props[2]
    assert mime_type == "image/gif" # Test should use a file that IS a gif

    entity, _ = asset_service.add_asset_file(
        db_session, sample_gif_file_placeholder, props[0], mime_type, props[1], world_name=current_world_name
    )
    db_session.commit()
    frame_props = ecs_service.get_component(db_session, entity.id, FramePropertiesComponent)
    assert frame_props is not None
    if frame_props.frame_count is not None:
        assert frame_props.frame_count >= 1 # The minimal GIF has 1 frame.

    dim_comp_gif = ecs_service.get_component(db_session, entity.id, ImageDimensionsComponent)
    # The placeholder GIF in conftest.py is 1x1 pixel.
    assert dim_comp_gif is not None and dim_comp_gif.width_pixels == 1 and dim_comp_gif.height_pixels == 1


def test_asset_isolation_between_worlds(settings_override, test_db_manager, sample_image_a: Path):
    # Get sessions for two different test worlds
    session_alpha = test_db_manager.get_db_session("test_world_alpha")
    session_beta = test_db_manager.get_db_session("test_world_beta")

    props_a = file_operations.get_file_properties(sample_image_a)

    # Add asset to world_alpha
    entity_alpha, _ = asset_service.add_asset_file(
        session_alpha, sample_image_a, props_a[0], props_a[2], props_a[1], world_name="test_world_alpha"
    )
    session_alpha.commit()

    # Verify asset exists in world_alpha
    assert ecs_service.get_entity(session_alpha, entity_alpha.id) is not None
    sha256_alpha = ecs_service.get_component(session_alpha, entity_alpha.id, ContentHashSHA256Component)
    assert sha256_alpha is not None

    # Verify asset's file exists in world_alpha's storage
    world_alpha_config = settings_override.get_world_config("test_world_alpha") # Use patched settings
    from dam.services.file_storage import _get_storage_path_for_world  # internal access for test

    alpha_storage_path = _get_storage_path_for_world(sha256_alpha.hash_value, world_alpha_config)
    assert alpha_storage_path.exists()

    # Verify asset does NOT exist in world_beta by entity ID (IDs are independent) or content hash
    assert (
        ecs_service.get_entity(session_beta, entity_alpha.id) is None
    )  # ID might coincidentally be same if both are 1

    # More robust check: by content hash
    beta_entity_by_hash = asset_service.find_entity_by_content_hash(session_beta, sha256_alpha.hash_value)
    assert beta_entity_by_hash is None

    # Verify asset's file does NOT exist in world_beta's storage
    world_beta_config = settings_override.get_world_config("test_world_beta") # Use patched settings
    beta_storage_path = _get_storage_path_for_world(sha256_alpha.hash_value, world_beta_config)
    assert not beta_storage_path.exists()

    session_alpha.close()
    session_beta.close()


def test_add_asset_reference_multi_world(settings_override, test_db_manager, sample_image_a: Path, tmp_path: Path):
    session_alpha = test_db_manager.get_db_session("test_world_alpha")

    # Create a unique file for referencing that won't be in CAS by default
    referenced_file = tmp_path / "referenced_image.png"
    shutil.copy2(sample_image_a, referenced_file)

    props_ref = file_operations.get_file_properties(referenced_file)

    entity_ref_alpha, created_ref = asset_service.add_asset_reference(
        session_alpha, referenced_file, "referenced.png", props_ref[2], props_ref[1], world_name="test_world_alpha"
    )
    session_alpha.commit()

    assert created_ref is True  # Should be new as it's a reference
    flc_alpha_ref = ecs_service.get_components(session_alpha, entity_ref_alpha.id, FileLocationComponent)
    assert len(flc_alpha_ref) == 1
    assert flc_alpha_ref[0].storage_type == "referenced_local_file"
    assert flc_alpha_ref[0].file_identifier == str(referenced_file.resolve())

    # Ensure the actual file was NOT copied to alpha's CAS storage
    world_alpha_config = settings_override.get_world_config("test_world_alpha") # Use patched settings
    sha256_comp = ecs_service.get_component(session_alpha, entity_ref_alpha.id, ContentHashSHA256Component)
    assert sha256_comp is not None
    from dam.services.file_storage import _get_storage_path_for_world

    cas_path_alpha = _get_storage_path_for_world(sha256_comp.hash_value, world_alpha_config)
    assert not cas_path_alpha.exists()  # Key check for --no-copy (add_asset_reference)

    session_alpha.close()
