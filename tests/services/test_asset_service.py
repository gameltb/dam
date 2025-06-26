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


# Helper to initialize scheduler for tests
def get_test_scheduler() -> WorldScheduler:
    resource_mgr = ResourceManager()
    resource_mgr.add_resource(FileOperationsResource()) # Add necessary resources
    return WorldScheduler(resource_mgr)


@pytest.mark.asyncio
async def test_add_image_asset_creates_perceptual_hashes(settings_override, db_session: Session, sample_image_a: Path):
    current_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config = settings_override.get_world_config(current_world_name)
    scheduler = get_test_scheduler()

    try:
        import imagehash # noqa: F401
    except ImportError:
        pytest.skip("ImageHash not installed.")

    props = file_operations.get_file_properties(sample_image_a)
    original_filename, size_bytes, mime_type = props[0], props[1], props[2]

    # Check if entity exists before (to determine if new one was created)
    image_sha256_hash = file_operations.calculate_sha256(sample_image_a)
    entity_before = asset_service.find_entity_by_content_hash(db_session, image_sha256_hash, "sha256")

    event = AssetFileIngestionRequested(
        filepath_on_disk=sample_image_a,
        original_filename=original_filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        world_name=current_world_name,
    )
    world_context = WorldContext(session=db_session, world_name=current_world_name, world_config=world_config)
    await scheduler.dispatch_event(event, world_context)
    # dispatch_event now commits session internally on success.

    entity = asset_service.find_entity_by_content_hash(db_session, image_sha256_hash, "sha256")
    assert entity is not None
    created_new = entity_before is None
    assert created_new is True # For this test, assuming it's a new asset

    phashes = ecs_service.get_components(db_session, entity.id, ImagePerceptualPHashComponent)
    ahashes = ecs_service.get_components(db_session, entity.id, ImagePerceptualAHashComponent)
    dhashes = ecs_service.get_components(db_session, entity.id, ImagePerceptualDHashComponent)
    assert len(phashes) + len(ahashes) + len(dhashes) > 0, "Perceptual hash components should be created by event handler"

    marker_comp = ecs_service.get_component(db_session, entity.id, NeedsMetadataExtractionComponent)
    assert marker_comp is not None, "Entity should be marked with NeedsMetadataExtractionComponent"

    # Run metadata stage to check for ImageDimensionsComponent (if this test implies it)
    # For this test, the original only checked marker, so let's stick to that for now.
    # If ImageDimensionsComponent needs to be checked, uncomment and adapt:
    # await scheduler.execute_stage(SystemStage.METADATA_EXTRACTION, world_context)
    # dim_comp = ecs_service.get_component(db_session, entity.id, ImageDimensionsComponent)
    # assert dim_comp is not None

    sha256_comp = ecs_service.get_component(db_session, entity.id, ContentHashSHA256Component)
    assert sha256_comp is not None

    from dam.services.file_storage import _get_storage_path_for_world
    reconstructed_path = _get_storage_path_for_world(sha256_comp.hash_value, world_config)
    assert reconstructed_path.exists()
    expected_file_path_fragment = (
        Path(world_config.ASSET_STORAGE_PATH)
        / sha256_comp.hash_value[:2]
        / sha256_comp.hash_value[2:4]
        / sha256_comp.hash_value
    )
    assert str(expected_file_path_fragment) in str(reconstructed_path)


@pytest.mark.asyncio
async def test_add_non_image_asset_no_perceptual_hashes(settings_override, db_session: Session, sample_text_file: Path):
    current_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config = settings_override.get_world_config(current_world_name)
    scheduler = get_test_scheduler()

    props = file_operations.get_file_properties(sample_text_file)
    original_filename, size_bytes, mime_type = props[0], props[1], props[2]

    text_sha256_hash = file_operations.calculate_sha256(sample_text_file)
    entity_before = asset_service.find_entity_by_content_hash(db_session, text_sha256_hash, "sha256")

    event = AssetFileIngestionRequested(
        filepath_on_disk=sample_text_file,
        original_filename=original_filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        world_name=current_world_name,
    )
    world_context = WorldContext(session=db_session, world_name=current_world_name, world_config=world_config)
    await scheduler.dispatch_event(event, world_context)

    entity = asset_service.find_entity_by_content_hash(db_session, text_sha256_hash, "sha256")
    assert entity is not None
    created_new = entity_before is None
    assert created_new is True

    assert len(ecs_service.get_components(db_session, entity.id, ImagePerceptualPHashComponent)) == 0


@pytest.mark.asyncio
async def test_add_existing_image_content_adds_missing_hashes(
    settings_override, db_session: Session, sample_image_a: Path, tmp_path: Path
):
    current_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config = settings_override.get_world_config(current_world_name)
    scheduler = get_test_scheduler()

    try:
        import imagehash # noqa: F401
    except ImportError:
        pytest.skip("ImageHash or Pillow not installed.")

    props1 = file_operations.get_file_properties(sample_image_a)
    event1 = AssetFileIngestionRequested(
        sample_image_a, props1[0], props1[2], props1[1], current_world_name
    )
    world_context = WorldContext(db_session, current_world_name, world_config)
    await scheduler.dispatch_event(event1, world_context)

    image_sha256_hash = file_operations.calculate_sha256(sample_image_a)
    entity1 = asset_service.find_entity_by_content_hash(db_session, image_sha256_hash, "sha256")
    assert entity1 is not None

    initial_dhashes = ecs_service.get_components(db_session, entity1.id, ImagePerceptualDHashComponent)
    if not initial_dhashes:
        pytest.skip("Initial event dispatch did not generate dhash.")

    # Manually remove the dhash to simulate it being missing
    db_session.delete(initial_dhashes[0])
    db_session.commit() # Commit removal
    assert not ecs_service.get_components(db_session, entity1.id, ImagePerceptualDHashComponent)

    copy_of_sample_image_a = tmp_path / "sample_A_copy.png"
    shutil.copy2(sample_image_a, copy_of_sample_image_a)
    props2 = file_operations.get_file_properties(copy_of_sample_image_a)

    # Sanity check: entity should exist before this event
    entity_before_event2 = asset_service.find_entity_by_content_hash(db_session, image_sha256_hash, "sha256")
    assert entity_before_event2 is not None

    event2 = AssetFileIngestionRequested(
        copy_of_sample_image_a, props2[0], props2[2], props2[1], current_world_name
    )
    # Use a new world_context for the second event if dispatch_event closes/invalidates the session
    # However, current dispatch_event commits but leaves session usable.
    await scheduler.dispatch_event(event2, world_context)

    entity2 = asset_service.find_entity_by_content_hash(db_session, image_sha256_hash, "sha256")
    assert entity2 is not None
    assert entity2.id == entity1.id # Should be the same entity

    final_dhashes = ecs_service.get_components(db_session, entity2.id, ImagePerceptualDHashComponent)
    assert len(final_dhashes) > 0
    assert final_dhashes[0].hash_value == initial_dhashes[0].hash_value


@pytest.mark.asyncio
async def test_add_video_asset_marks_for_metadata_extraction(
    settings_override, db_session: Session, sample_video_file_placeholder: Path
):
    current_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config = settings_override.get_world_config(current_world_name)
    scheduler = get_test_scheduler()

    try:
        from hachoir.parser import createParser # noqa
    except ImportError:
        pytest.skip("Hachoir not installed for metadata systems (though not directly used by ingestion event).")

    props = file_operations.get_file_properties(sample_video_file_placeholder)
    mime_type = "video/mp4" if props[2] == "application/octet-stream" else props[2]

    video_sha256_hash = file_operations.calculate_sha256(sample_video_file_placeholder)

    event = AssetFileIngestionRequested(
        sample_video_file_placeholder, props[0], mime_type, props[1], current_world_name
    )
    world_context = WorldContext(db_session, current_world_name, world_config)
    await scheduler.dispatch_event(event, world_context)

    entity = asset_service.find_entity_by_content_hash(db_session, video_sha256_hash, "sha256")
    assert entity is not None

    assert ecs_service.get_component(db_session, entity.id, ImageDimensionsComponent) is None
    assert ecs_service.get_component(db_session, entity.id, FramePropertiesComponent) is None
    assert ecs_service.get_component(db_session, entity.id, AudioPropertiesComponent) is None

    marker_comp = ecs_service.get_component(db_session, entity.id, NeedsMetadataExtractionComponent)
    assert marker_comp is not None, "Video asset should be marked with NeedsMetadataExtractionComponent"

    # To fully test, run metadata stage and check for components
    await scheduler.execute_stage(SystemStage.METADATA_EXTRACTION, world_context)
    # Now check if components were added (assuming sample_video_file_placeholder is a valid video)
    # This part depends on the actual video file and hachoir's ability to parse it.
    # For a placeholder, these might still be None. If it's a real video, they should exist.
    # For example:
    # if sample_video_file_placeholder.name != "empty.mp4": # Assuming "empty.mp4" is a dummy
    #     assert ecs_service.get_component(db_session, entity.id, ImageDimensionsComponent) is not None
    #     assert ecs_service.get_component(db_session, entity.id, FramePropertiesComponent) is not None


@pytest.mark.asyncio
async def test_add_audio_asset_marks_for_metadata_extraction(
    settings_override, db_session: Session, sample_audio_file_placeholder: Path
):
    current_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config = settings_override.get_world_config(current_world_name)
    scheduler = get_test_scheduler()

    try:
        from hachoir.parser import createParser # noqa
    except ImportError:
        pytest.skip("Hachoir not installed.")

    props = file_operations.get_file_properties(sample_audio_file_placeholder)
    mime_type = "audio/mpeg" if props[2] == "application/octet-stream" else props[2]
    audio_sha256_hash = file_operations.calculate_sha256(sample_audio_file_placeholder)

    event = AssetFileIngestionRequested(
        sample_audio_file_placeholder, props[0], mime_type, props[1], current_world_name
    )
    world_context = WorldContext(db_session, current_world_name, world_config)
    await scheduler.dispatch_event(event, world_context)

    entity = asset_service.find_entity_by_content_hash(db_session, audio_sha256_hash, "sha256")
    assert entity is not None

    assert ecs_service.get_component(db_session, entity.id, AudioPropertiesComponent) is None
    marker_comp = ecs_service.get_component(db_session, entity.id, NeedsMetadataExtractionComponent)
    assert marker_comp is not None, "Audio asset should be marked with NeedsMetadataExtractionComponent"

    await scheduler.execute_stage(SystemStage.METADATA_EXTRACTION, world_context)
    # if sample_audio_file_placeholder.name != "empty.mp3": # Assuming "empty.mp3" is a dummy
    #    assert ecs_service.get_component(db_session, entity.id, AudioPropertiesComponent) is not None


@pytest.mark.asyncio
async def test_add_gif_asset_marks_for_metadata_extraction(
    settings_override, db_session: Session, sample_gif_file_placeholder: Path
):
    current_world_name = settings_override.DEFAULT_WORLD_NAME
    world_config = settings_override.get_world_config(current_world_name)
    scheduler = get_test_scheduler()

    props = file_operations.get_file_properties(sample_gif_file_placeholder)
    mime_type = "image/gif" if props[2] != "image/gif" else props[2]
    assert mime_type == "image/gif"
    gif_sha256_hash = file_operations.calculate_sha256(sample_gif_file_placeholder)

    event = AssetFileIngestionRequested(
        sample_gif_file_placeholder, props[0], mime_type, props[1], current_world_name
    )
    world_context = WorldContext(db_session, current_world_name, world_config)
    await scheduler.dispatch_event(event, world_context)

    entity = asset_service.find_entity_by_content_hash(db_session, gif_sha256_hash, "sha256")
    assert entity is not None

    assert ecs_service.get_component(db_session, entity.id, FramePropertiesComponent) is None
    assert ecs_service.get_component(db_session, entity.id, ImageDimensionsComponent) is None
    marker_comp = ecs_service.get_component(db_session, entity.id, NeedsMetadataExtractionComponent)
    assert marker_comp is not None, "GIF asset should be marked with NeedsMetadataExtractionComponent"

    await scheduler.execute_stage(SystemStage.METADATA_EXTRACTION, world_context)
    # if sample_gif_file_placeholder.name != "empty.gif": # Assuming "empty.gif" is a dummy
    #     assert ecs_service.get_component(db_session, entity.id, ImageDimensionsComponent) is not None
    #     # FramePropertiesComponent might depend on specific GIF content (animated or not)
    #     # For a simple static GIF, FramePropertiesComponent might not be added or have frame_count=1
    #     fp_comp = ecs_service.get_component(db_session, entity.id, FramePropertiesComponent)
    #     assert fp_comp is not None
    #     assert fp_comp.frame_count >= 1


@pytest.mark.asyncio
async def test_asset_isolation_between_worlds(settings_override, test_db_manager, sample_image_a: Path):
    scheduler_alpha = get_test_scheduler()
    # scheduler_beta = get_test_scheduler() # Schedulers are stateless, can reuse

    world_alpha_name = "test_world_alpha"
    world_beta_name = "test_world_beta"

    session_alpha = test_db_manager.get_db_session(world_alpha_name)
    session_beta = test_db_manager.get_db_session(world_beta_name)

    world_alpha_config = settings_override.get_world_config(world_alpha_name)
    world_beta_config = settings_override.get_world_config(world_beta_name)

    props_a = file_operations.get_file_properties(sample_image_a)
    image_sha256_hash = file_operations.calculate_sha256(sample_image_a)

    event_alpha = AssetFileIngestionRequested(
        sample_image_a, props_a[0], props_a[2], props_a[1], world_alpha_name
    )
    wc_alpha = WorldContext(session_alpha, world_alpha_name, world_alpha_config)
    await scheduler_alpha.dispatch_event(event_alpha, wc_alpha)

    entity_alpha = asset_service.find_entity_by_content_hash(session_alpha, image_sha256_hash)
    assert entity_alpha is not None
    sha256_alpha_comp = ecs_service.get_component(session_alpha, entity_alpha.id, ContentHashSHA256Component)
    assert sha256_alpha_comp is not None

    from dam.services.file_storage import _get_storage_path_for_world
    alpha_storage_path = _get_storage_path_for_world(sha256_alpha_comp.hash_value, world_alpha_config)
    assert alpha_storage_path.exists()

    beta_entity_by_hash = asset_service.find_entity_by_content_hash(session_beta, sha256_alpha_comp.hash_value)
    assert beta_entity_by_hash is None

    beta_storage_path = _get_storage_path_for_world(sha256_alpha_comp.hash_value, world_beta_config)
    assert not beta_storage_path.exists()

    session_alpha.close()
    session_beta.close()


@pytest.mark.asyncio
async def test_add_asset_reference_multi_world(settings_override, test_db_manager, sample_image_a: Path, tmp_path: Path):
    world_alpha_name = "test_world_alpha"
    session_alpha = test_db_manager.get_db_session(world_alpha_name)
    world_alpha_config = settings_override.get_world_config(world_alpha_name)
    scheduler = get_test_scheduler()

    referenced_file = tmp_path / "referenced_image.png"
    shutil.copy2(sample_image_a, referenced_file)
    props_ref = file_operations.get_file_properties(referenced_file)
    ref_sha256_hash = file_operations.calculate_sha256(referenced_file)

    event_ref_alpha = AssetReferenceIngestionRequested(
        referenced_file, "referenced.png", props_ref[2], props_ref[1], world_alpha_name
    )
    wc_alpha = WorldContext(session_alpha, world_alpha_name, world_alpha_config)
    await scheduler.dispatch_event(event_ref_alpha, wc_alpha)

    entity_ref_alpha = asset_service.find_entity_by_content_hash(session_alpha, ref_sha256_hash)
    assert entity_ref_alpha is not None

    flc_alpha_ref_list = ecs_service.get_components(session_alpha, entity_ref_alpha.id, FileLocationComponent)
    assert len(flc_alpha_ref_list) == 1
    flc_alpha_ref = flc_alpha_ref_list[0]
    assert flc_alpha_ref.storage_type == "local_reference"
    assert flc_alpha_ref.physical_path_or_key == str(referenced_file.resolve())

    sha256_comp = ecs_service.get_component(session_alpha, entity_ref_alpha.id, ContentHashSHA256Component)
    assert sha256_comp is not None

    from dam.services.file_storage import _get_storage_path_for_world
    cas_path_alpha = _get_storage_path_for_world(sha256_comp.hash_value, world_alpha_config)
    assert not cas_path_alpha.exists()

    session_alpha.close()
