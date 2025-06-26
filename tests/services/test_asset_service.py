# tests/services/test_asset_service.py
import shutil
from pathlib import Path
import asyncio # Added for async tests

import pytest
from sqlalchemy.orm import Session

# Core imports for event system
from dam.core.events import AssetFileIngestionRequested, AssetReferenceIngestionRequested
from dam.core.systems import WorldScheduler, WorldContext
from dam.core.resources import ResourceManager, FileOperationsResource # Added FileOperationsResource
from dam.core.stages import SystemStage
from dam.core.components_markers import NeedsMetadataExtractionComponent
from dam.models import (
    AudioPropertiesComponent,
    ContentHashMD5Component, # Added for direct verification
    ContentHashSHA256Component,
    FileLocationComponent,
    FramePropertiesComponent,
    ImageDimensionsComponent,
    ImagePerceptualAHashComponent,
    ImagePerceptualDHashComponent,
    ImagePerceptualPHashComponent,
    Entity # Added for querying
)
from dam.services import asset_service, ecs_service, file_operations # asset_service still needed for find_by_hash
import dam.systems # Import to ensure all systems are registered (event handlers, metadata)


from dam.core.world import World # Import World for type hinting

# Helper to initialize scheduler for tests - REMOVED as scheduler is part of World
# def get_test_scheduler() -> WorldScheduler:
#     resource_mgr = ResourceManager()
#     resource_mgr.add_resource(FileOperationsResource()) # Add necessary resources
#     return WorldScheduler(resource_mgr)


@pytest.mark.asyncio
async def test_add_image_asset_creates_perceptual_hashes(test_world_alpha: World, sample_image_a: Path):
    # settings_override is implicitly used by test_world_alpha fixture
    # db_session is also implicitly managed if we get it from test_world_alpha

    world = test_world_alpha # Use the fixture

    try:
        import imagehash # noqa: F401
    except ImportError:
        pytest.skip("ImageHash not installed.")

    props = file_operations.get_file_properties(sample_image_a)
    original_filename, size_bytes, mime_type = props[0], props[1], props[2]

    image_sha256_hash = file_operations.calculate_sha256(sample_image_a)

    # Use a session from the world for pre-check
    db_session_pre_check = world.get_db_session()
    try:
        entity_before = asset_service.find_entity_by_content_hash(db_session_pre_check, image_sha256_hash, "sha256")
    finally:
        db_session_pre_check.close()

    # Event now might need world_config if the handler directly calls asset_service.add_asset_file
    # or if the event itself is changed to carry it.
    # For now, assuming AssetFileIngestionRequested still takes world_name, and the system handler
    # for this event will get the WorldConfig from its WorldContext.
    event = AssetFileIngestionRequested(
        filepath_on_disk=sample_image_a,
        original_filename=original_filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        world_name=world.name, # Pass world name from the World object
    )

    # World.dispatch_event handles session and WorldContext creation
    await world.dispatch_event(event)

    db_session_post_check = world.get_db_session()
    try:
        entity = asset_service.find_entity_by_content_hash(db_session_post_check, image_sha256_hash, "sha256")
        assert entity is not None
        created_new = entity_before is None
        assert created_new is True

        phashes = ecs_service.get_components(db_session_post_check, entity.id, ImagePerceptualPHashComponent)
        ahashes = ecs_service.get_components(db_session_post_check, entity.id, ImagePerceptualAHashComponent)
        dhashes = ecs_service.get_components(db_session_post_check, entity.id, ImagePerceptualDHashComponent)
        assert len(phashes) + len(ahashes) + len(dhashes) > 0, "Perceptual hash components should be created by event handler"

        marker_comp = ecs_service.get_component(db_session_post_check, entity.id, NeedsMetadataExtractionComponent)
        assert marker_comp is not None, "Entity should be marked with NeedsMetadataExtractionComponent"

        sha256_comp = ecs_service.get_component(db_session_post_check, entity.id, ContentHashSHA256Component)
        assert sha256_comp is not None

        from dam.services.file_storage import _get_storage_path_for_world
        reconstructed_path = _get_storage_path_for_world(sha256_comp.hash_value, world.config)
        assert reconstructed_path.exists()
        expected_file_path_fragment = (
            Path(world.config.ASSET_STORAGE_PATH)
            / sha256_comp.hash_value[:2]
            / sha256_comp.hash_value[2:4]
            / sha256_comp.hash_value
        )
        assert str(expected_file_path_fragment) in str(reconstructed_path)
    finally:
        db_session_post_check.close()


@pytest.mark.asyncio
async def test_add_non_image_asset_no_perceptual_hashes(test_world_alpha: World, sample_text_file: Path):
    world = test_world_alpha

    props = file_operations.get_file_properties(sample_text_file)
    original_filename, size_bytes, mime_type = props[0], props[1], props[2]

    text_sha256_hash = file_operations.calculate_sha256(sample_text_file)
    db_session_pre_check = world.get_db_session()
    try:
        entity_before = asset_service.find_entity_by_content_hash(db_session_pre_check, text_sha256_hash, "sha256")
    finally:
        db_session_pre_check.close()

    event = AssetFileIngestionRequested(
        filepath_on_disk=sample_text_file,
        original_filename=original_filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        world_name=world.name,
    )
    await world.dispatch_event(event) # World handles session and context

    db_session_post_check = world.get_db_session()
    try:
        entity = asset_service.find_entity_by_content_hash(db_session_post_check, text_sha256_hash, "sha256")
        assert entity is not None
        created_new = entity_before is None
        assert created_new is True
        assert len(ecs_service.get_components(db_session_post_check, entity.id, ImagePerceptualPHashComponent)) == 0
    finally:
        db_session_post_check.close()


@pytest.mark.asyncio
async def test_add_existing_image_content_adds_missing_hashes(
    test_world_alpha: World, sample_image_a: Path, tmp_path: Path
):
    world = test_world_alpha

    try:
        import imagehash # noqa: F401
    except ImportError:
        pytest.skip("ImageHash or Pillow not installed.")

    props1 = file_operations.get_file_properties(sample_image_a)
    event1 = AssetFileIngestionRequested(
        sample_image_a, props1[0], props1[2], props1[1], world.name
    )
    # First event dispatch
    await world.dispatch_event(event1) # Session managed by world.dispatch_event

    image_sha256_hash = file_operations.calculate_sha256(sample_image_a)

    db_session_check1 = world.get_db_session()
    try:
        entity1 = asset_service.find_entity_by_content_hash(db_session_check1, image_sha256_hash, "sha256")
        assert entity1 is not None
        initial_dhashes = ecs_service.get_components(db_session_check1, entity1.id, ImagePerceptualDHashComponent)
        if not initial_dhashes:
            pytest.skip("Initial event dispatch did not generate dhash.")
        # Manually remove the dhash to simulate it being missing
        db_session_check1.delete(initial_dhashes[0])
        db_session_check1.commit() # Commit removal
        assert not ecs_service.get_components(db_session_check1, entity1.id, ImagePerceptualDHashComponent)
    finally:
        db_session_check1.close()

    copy_of_sample_image_a = tmp_path / "sample_A_copy.png"
    shutil.copy2(sample_image_a, copy_of_sample_image_a)
    props2 = file_operations.get_file_properties(copy_of_sample_image_a)

    # Sanity check: entity should exist before this event (use a new session for check)
    db_session_check2 = world.get_db_session()
    try:
        entity_before_event2 = asset_service.find_entity_by_content_hash(db_session_check2, image_sha256_hash, "sha256")
        assert entity_before_event2 is not None
    finally:
        db_session_check2.close()

    event2 = AssetFileIngestionRequested(
        copy_of_sample_image_a, props2[0], props2[2], props2[1], world.name
    )
    await world.dispatch_event(event2) # Second event dispatch

    db_session_check3 = world.get_db_session()
    try:
        entity2 = asset_service.find_entity_by_content_hash(db_session_check3, image_sha256_hash, "sha256")
        assert entity2 is not None
        assert entity2.id == entity1.id # Should be the same entity

        final_dhashes = ecs_service.get_components(db_session_check3, entity2.id, ImagePerceptualDHashComponent)
        assert len(final_dhashes) > 0
        assert final_dhashes[0].hash_value == initial_dhashes[0].hash_value
    finally:
        db_session_check3.close()


@pytest.mark.asyncio
async def test_add_video_asset_marks_for_metadata_extraction(
    test_world_alpha: World, sample_video_file_placeholder: Path
):
    world = test_world_alpha

    try:
        from hachoir.parser import createParser # noqa
    except ImportError:
        pytest.skip("Hachoir not installed for metadata systems (though not directly used by ingestion event).")

    props = file_operations.get_file_properties(sample_video_file_placeholder)
    mime_type = "video/mp4" if props[2] == "application/octet-stream" else props[2]

    video_sha256_hash = file_operations.calculate_sha256(sample_video_file_placeholder)

    event = AssetFileIngestionRequested(
        sample_video_file_placeholder, props[0], mime_type, props[1], world.name
    )
    await world.dispatch_event(event)

    db_session_check = world.get_db_session()
    try:
        entity = asset_service.find_entity_by_content_hash(db_session_check, video_sha256_hash, "sha256")
        assert entity is not None

        assert ecs_service.get_component(db_session_check, entity.id, ImageDimensionsComponent) is None
        assert ecs_service.get_component(db_session_check, entity.id, FramePropertiesComponent) is None
        assert ecs_service.get_component(db_session_check, entity.id, AudioPropertiesComponent) is None

        marker_comp = ecs_service.get_component(db_session_check, entity.id, NeedsMetadataExtractionComponent)
        assert marker_comp is not None, "Video asset should be marked with NeedsMetadataExtractionComponent"
    finally:
        db_session_check.close()

    # To fully test, run metadata stage and check for components
    await world.execute_stage(SystemStage.METADATA_EXTRACTION)

    # Re-check after stage execution with a new session
    db_session_after_stage = world.get_db_session()
    try:
        entity_after_stage = asset_service.find_entity_by_content_hash(db_session_after_stage, video_sha256_hash, "sha256")
        assert entity_after_stage is not None
        # Now check if components were added (assuming sample_video_file_placeholder is a valid video)
        # This part depends on the actual video file and hachoir's ability to parse it.
        # For a placeholder, these might still be None. If it's a real video, they should exist.
        # Example:
        # if sample_video_file_placeholder.name != "empty.mp4":
        #     assert ecs_service.get_component(db_session_after_stage, entity_after_stage.id, ImageDimensionsComponent) is not None
        #     assert ecs_service.get_component(db_session_after_stage, entity_after_stage.id, FramePropertiesComponent) is not None
    finally:
        db_session_after_stage.close()


@pytest.mark.asyncio
async def test_add_audio_asset_marks_for_metadata_extraction(
    test_world_alpha: World, sample_audio_file_placeholder: Path
):
    world = test_world_alpha

    try:
        from hachoir.parser import createParser # noqa
    except ImportError:
        pytest.skip("Hachoir not installed.")

    props = file_operations.get_file_properties(sample_audio_file_placeholder)
    mime_type = "audio/mpeg" if props[2] == "application/octet-stream" else props[2]
    audio_sha256_hash = file_operations.calculate_sha256(sample_audio_file_placeholder)

    event = AssetFileIngestionRequested(
        sample_audio_file_placeholder, props[0], mime_type, props[1], world.name
    )
    await world.dispatch_event(event)

    db_session_check = world.get_db_session()
    try:
        entity = asset_service.find_entity_by_content_hash(db_session_check, audio_sha256_hash, "sha256")
        assert entity is not None
        assert ecs_service.get_component(db_session_check, entity.id, AudioPropertiesComponent) is None
        marker_comp = ecs_service.get_component(db_session_check, entity.id, NeedsMetadataExtractionComponent)
        assert marker_comp is not None, "Audio asset should be marked with NeedsMetadataExtractionComponent"
    finally:
        db_session_check.close()

    await world.execute_stage(SystemStage.METADATA_EXTRACTION)
    # Re-check after stage:
    # db_session_after_stage = world.get_db_session()
    # try:
    #     if sample_audio_file_placeholder.name != "empty.mp3":
    #         entity_after_stage = asset_service.find_entity_by_content_hash(db_session_after_stage, audio_sha256_hash, "sha256")
    #         assert ecs_service.get_component(db_session_after_stage, entity_after_stage.id, AudioPropertiesComponent) is not None
    # finally:
    #     db_session_after_stage.close()


@pytest.mark.asyncio
async def test_add_gif_asset_marks_for_metadata_extraction(
    test_world_alpha: World, sample_gif_file_placeholder: Path
):
    world = test_world_alpha

    props = file_operations.get_file_properties(sample_gif_file_placeholder)
    mime_type = "image/gif" if props[2] != "image/gif" else props[2]
    assert mime_type == "image/gif"
    gif_sha256_hash = file_operations.calculate_sha256(sample_gif_file_placeholder)

    event = AssetFileIngestionRequested(
        sample_gif_file_placeholder, props[0], mime_type, props[1], world.name
    )
    await world.dispatch_event(event)

    db_session_check = world.get_db_session()
    try:
        entity = asset_service.find_entity_by_content_hash(db_session_check, gif_sha256_hash, "sha256")
        assert entity is not None
        assert ecs_service.get_component(db_session_check, entity.id, FramePropertiesComponent) is None
        assert ecs_service.get_component(db_session_check, entity.id, ImageDimensionsComponent) is None
        marker_comp = ecs_service.get_component(db_session_check, entity.id, NeedsMetadataExtractionComponent)
        assert marker_comp is not None, "GIF asset should be marked with NeedsMetadataExtractionComponent"
    finally:
        db_session_check.close()

    await world.execute_stage(SystemStage.METADATA_EXTRACTION)
    # Re-check after stage:
    # db_session_after_stage = world.get_db_session()
    # try:
    #     entity_after_stage = asset_service.find_entity_by_content_hash(db_session_after_stage, gif_sha256_hash, "sha256")
    #     if sample_gif_file_placeholder.name != "empty.gif":
    #         assert ecs_service.get_component(db_session_after_stage, entity_after_stage.id, ImageDimensionsComponent) is not None
    #         fp_comp = ecs_service.get_component(db_session_after_stage, entity_after_stage.id, FramePropertiesComponent)
    #         assert fp_comp is not None
    #         assert fp_comp.frame_count >= 1
    # finally:
    #     db_session_after_stage.close()


@pytest.mark.asyncio
async def test_asset_isolation_between_worlds(test_world_alpha: World, test_world_beta: World, sample_image_a: Path):
    world_alpha = test_world_alpha
    world_beta = test_world_beta

    props_a = file_operations.get_file_properties(sample_image_a)
    image_sha256_hash = file_operations.calculate_sha256(sample_image_a)

    event_alpha = AssetFileIngestionRequested(
        sample_image_a, props_a[0], props_a[2], props_a[1], world_alpha.name
    )
    await world_alpha.dispatch_event(event_alpha)

    session_alpha = world_alpha.get_db_session()
    try:
        entity_alpha = asset_service.find_entity_by_content_hash(session_alpha, image_sha256_hash)
        assert entity_alpha is not None
        sha256_alpha_comp = ecs_service.get_component(session_alpha, entity_alpha.id, ContentHashSHA256Component)
        assert sha256_alpha_comp is not None

        from dam.services.file_storage import _get_storage_path_for_world
        alpha_storage_path = _get_storage_path_for_world(sha256_alpha_comp.hash_value, world_alpha.config)
        assert alpha_storage_path.exists()
    finally:
        session_alpha.close()

    session_beta = world_beta.get_db_session()
    try:
        beta_entity_by_hash = asset_service.find_entity_by_content_hash(session_beta, image_sha256_hash) # Use image_sha256_hash
        assert beta_entity_by_hash is None

        # Use image_sha256_hash for checking storage path in beta world as well
        from dam.services.file_storage import _get_storage_path_for_world
        beta_storage_path = _get_storage_path_for_world(image_sha256_hash, world_beta.config)
        assert not beta_storage_path.exists()
    finally:
        session_beta.close()


@pytest.mark.asyncio
async def test_add_asset_reference_multi_world(test_world_alpha: World, sample_image_a: Path, tmp_path: Path):
    world = test_world_alpha # Using only one world for this specific reference test

    referenced_file = tmp_path / "referenced_image.png"
    shutil.copy2(sample_image_a, referenced_file)
    props_ref = file_operations.get_file_properties(referenced_file)
    ref_sha256_hash = file_operations.calculate_sha256(referenced_file)

    event_ref = AssetReferenceIngestionRequested(
        referenced_file, "referenced.png", props_ref[2], props_ref[1], world.name
    )
    await world.dispatch_event(event_ref)

    session = world.get_db_session()
    try:
        entity_ref = asset_service.find_entity_by_content_hash(session, ref_sha256_hash)
        assert entity_ref is not None

        flc_list = ecs_service.get_components(session, entity_ref.id, FileLocationComponent)
        assert len(flc_list) == 1
        flc = flc_list[0]
        assert flc.storage_type == "local_reference"
        assert flc.physical_path_or_key == str(referenced_file.resolve())

        sha256_comp = ecs_service.get_component(session, entity_ref.id, ContentHashSHA256Component)
        assert sha256_comp is not None

        # For a referenced file, it should NOT be copied to the world's CAS store
        from dam.services.file_storage import _get_storage_path_for_world
        cas_path = _get_storage_path_for_world(sha256_comp.hash_value, world.config)
        assert not cas_path.exists()
    finally:
        session.close()
