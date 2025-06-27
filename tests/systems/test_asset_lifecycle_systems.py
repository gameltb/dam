import pytest
import asyncio
from sqlalchemy.orm import Session

from dam.core.events import WebAssetIngestionRequested
from dam.core.world import World
from dam.models import Entity
from dam.models.original_source_info_component import OriginalSourceInfoComponent
from dam.models.web_source_component import WebSourceComponent
from dam.models.website_profile_component import WebsiteProfileComponent
from dam.services import ecs_service

# Assuming test_environment fixture from conftest.py or similar provides a configured World
# For now, let's adapt the test_environment or use a more direct world setup for system tests.
# The db_session fixture from conftest.py gives us a session, but systems operate within a World context.

# We need a way to get a test_world instance.
# Let's assume a fixture `test_world_with_db_session` that provides a World instance
# whose systems are registered and which uses the provided db_session.
# This might require enhancing conftest.py or using parts of the CLI test_environment logic.

# For simplicity in this step, we might manually create and register a world if a fixture isn't readily available.
# However, using a fixture similar to test_cli.py's test_environment is better.
# Let's assume `test_world` fixture is available from `tests/conftest.py` which gives a fully initialized world.

@pytest.mark.asyncio
async def test_handle_web_asset_ingestion_new_website(test_world_with_db_session: World):
    """
    Test handle_web_asset_ingestion_request when the website is new.
    It should create a Website Entity, its profile, an Asset Entity,
    and link them via WebSourceComponent.
    """
    world = test_world_with_db_session
    db_session = world.get_db_session() # Get session from the world

    website_url = "https://newgallery.example.com"
    asset_source_url = "https://newgallery.example.com/art/pic101"

    event_data = {
        "world_name": world.name,
        "website_identifier_url": website_url,
        "source_url": asset_source_url,
        "metadata_payload": {
            "website_name": "New Gallery", # Explicit name for profile
            "asset_title": "Sunset Bliss",
            "uploader_name": "photo_joe",
            "gallery_id": "pic101"
        },
        "tags": ["sunset", "nature"]
    }
    event = WebAssetIngestionRequested(**event_data)

    await world.dispatch_event(event)
    # world.dispatch_event commits the session internally

    # Verification
    # 1. Check Website Entity and Profile
    website_entities = ecs_service.find_entities_by_component_attribute_value(
        db_session, WebsiteProfileComponent, "main_url", website_url
    )
    assert len(website_entities) == 1
    website_entity = website_entities[0]

    profile_comp = ecs_service.get_component(db_session, website_entity.id, WebsiteProfileComponent)
    assert profile_comp is not None
    assert profile_comp.name == "New Gallery"
    assert profile_comp.main_url == website_url

    # 2. Check Asset Entity, OriginalSourceInfo, and WebSourceComponent
    # We need a way to find the asset entity. Let's assume it's the one with WebSourceComponent pointing to asset_source_url
    # This is a bit indirect. A better way might be if the event could return the created entity ID (future enhancement).

    # Query WebSourceComponent by source_url and website_entity_id
    stmt = db_session.query(WebSourceComponent).filter_by(source_url=asset_source_url, website_entity_id=website_entity.id)
    web_sources = stmt.all()

    assert len(web_sources) == 1
    web_source_comp = web_sources[0]
    asset_entity_id = web_source_comp.entity_id

    assert web_source_comp.asset_title == "Sunset Bliss"
    assert web_source_comp.uploader_name == "photo_joe"
    assert web_source_comp.gallery_id == "pic101"
    import json
    assert json.loads(web_source_comp.tags_json) == ["sunset", "nature"]

    osi_comp = ecs_service.get_component(db_session, asset_entity_id, OriginalSourceInfoComponent)
    assert osi_comp is not None
    assert osi_comp.source_type == "web_source"
    assert osi_comp.original_path == asset_source_url # As per current logic
    assert osi_comp.original_filename == "Sunset Bliss" # As derived


@pytest.mark.asyncio
async def test_handle_web_asset_ingestion_existing_website(test_world_with_db_session: World):
    """
    Test handle_web_asset_ingestion_request when the website entity already exists.
    It should use the existing Website Entity.
    """
    world = test_world_with_db_session
    db_session = world.get_db_session()

    # 1. Pre-create Website Entity
    website_url = "https://existinggallery.example.com"
    existing_website_entity = ecs_service.create_entity(db_session)
    ecs_service.add_component_to_entity(db_session, existing_website_entity.id, WebsiteProfileComponent(
        entity_id=existing_website_entity.id,
        name="Existing Gallery",
        main_url=website_url
    ))
    db_session.commit() # Commit pre-existing website

    # 2. Ingest web asset from this existing website
    asset_source_url = "https://existinggallery.example.com/art/pic202"
    event = WebAssetIngestionRequested(
        world_name=world.name,
        website_identifier_url=website_url,
        source_url=asset_source_url,
        metadata_payload={"asset_title": "Moonlit Path"}
    )
    await world.dispatch_event(event)

    # Verification
    # Check that no new WebsiteProfileComponent was created for this main_url
    website_profiles = db_session.query(WebsiteProfileComponent).filter_by(main_url=website_url).all()
    assert len(website_profiles) == 1
    assert website_profiles[0].entity_id == existing_website_entity.id # Ensure it's the same one

    # Check WebSourceComponent links to the existing website entity
    stmt = db_session.query(WebSourceComponent).filter_by(source_url=asset_source_url)
    web_source_comp = stmt.one_or_none() # Assuming source_url is unique enough for the test

    assert web_source_comp is not None
    assert web_source_comp.website_entity_id == existing_website_entity.id
    assert web_source_comp.asset_title == "Moonlit Path"


# Tests for file ingestion systems (adapted for byte hashes and source_type)

@pytest.mark.asyncio
async def test_handle_asset_file_ingestion_request_bytes_hashes(
    test_world_with_db_session: World, sample_text_file: Path, caplog
):
    """
    Test file ingestion system for correct byte hash storage and source_type.
    """
    world = test_world_with_db_session
    db_session = world.get_db_session()
    caplog.set_level(logging.INFO)

    from dam.core.events import AssetFileIngestionRequested
    from dam.services import file_operations
    import binascii
    from dam.models.content_hash_sha256_component import ContentHashSHA256Component
    from dam.models.content_hash_md5_component import ContentHashMD5Component

    props = file_operations.get_file_properties(sample_text_file)
    event = AssetFileIngestionRequested(
        filepath_on_disk=sample_text_file,
        original_filename=props[0],
        mime_type=props[2], # Mime type is at index 2
        size_bytes=props[1],  # Size is at index 1
        world_name=world.name
    )
    await world.dispatch_event(event)

    # Verification
    # Find entity - assume one entity created for this test
    # A more robust way would be to get entity_id if event processing could return it.
    # For now, query by original_filename and source_type from OriginalSourceInfoComponent.

    osi_comps = ecs_service.find_entities_by_component_attribute_value(
        db_session, OriginalSourceInfoComponent, "original_filename", props[0]
    )
    # Filter further if needed, e.g. by path, if filename isn't unique enough
    asset_entity = None
    for entity_candidate in osi_comps:
        osi = ecs_service.get_component(db_session, entity_candidate.id, OriginalSourceInfoComponent)
        if osi and osi.original_path == str(sample_text_file.resolve()):
            asset_entity = entity_candidate
            break

    assert asset_entity is not None, "Asset entity not found"

    # Check OriginalSourceInfoComponent
    osi_comp_retrieved = ecs_service.get_component(db_session, asset_entity.id, OriginalSourceInfoComponent)
    assert osi_comp_retrieved is not None
    assert osi_comp_retrieved.source_type == "local_file"

    # Check SHA256 hash (bytes)
    sha256_hex = file_operations.calculate_sha256(sample_text_file)
    sha256_bytes = binascii.unhexlify(sha256_hex)
    sha256_db_comp = ecs_service.get_component(db_session, asset_entity.id, ContentHashSHA256Component)
    assert sha256_db_comp is not None
    assert isinstance(sha256_db_comp.hash_value, bytes)
    assert sha256_db_comp.hash_value == sha256_bytes

    # Check MD5 hash (bytes)
    md5_hex = file_operations.calculate_md5(sample_text_file)
    md5_bytes = binascii.unhexlify(md5_hex)
    md5_db_comp = ecs_service.get_component(db_session, asset_entity.id, ContentHashMD5Component)
    assert md5_db_comp is not None
    assert isinstance(md5_db_comp.hash_value, bytes)
    assert md5_db_comp.hash_value == md5_bytes

    # Check log for successful processing
    assert f"Finished AssetFileIngestionRequested for Entity ID {asset_entity.id}" in caplog.text


@pytest.mark.asyncio
async def test_handle_asset_reference_ingestion_request_bytes_hashes(
    test_world_with_db_session: World, sample_image_file: Path, caplog # Using image for perceptual hashes
):
    """
    Test reference file ingestion system for correct byte hash storage, source_type,
    and perceptual hash (bytes) storage.
    """
    world = test_world_with_db_session
    db_session = world.get_db_session()
    caplog.set_level(logging.INFO)

    from dam.core.events import AssetReferenceIngestionRequested
    from dam.services import file_operations
    import binascii
    from dam.models.content_hash_sha256_component import ContentHashSHA256Component
    from dam.models.content_hash_md5_component import ContentHashMD5Component
    from dam.models.image_perceptual_hash_phash_component import ImagePerceptualPHashComponent

    props = file_operations.get_file_properties(sample_image_file)
    event = AssetReferenceIngestionRequested(
        filepath_on_disk=sample_image_file,
        original_filename=props[0],
        mime_type=props[2],
        size_bytes=props[1],
        world_name=world.name
    )
    await world.dispatch_event(event)

    # Verification (similar to file ingestion, find entity indirectly)
    osi_comps = ecs_service.find_entities_by_component_attribute_value(
        db_session, OriginalSourceInfoComponent, "original_filename", props[0]
    )
    asset_entity = None
    for entity_candidate in osi_comps:
        osi = ecs_service.get_component(db_session, entity_candidate.id, OriginalSourceInfoComponent)
        if osi and osi.original_path == str(sample_image_file.resolve()) and osi.source_type == "referenced_file":
            asset_entity = entity_candidate
            break
    assert asset_entity is not None, "Asset entity (referenced) not found"

    # Check OriginalSourceInfoComponent
    osi_comp_retrieved = ecs_service.get_component(db_session, asset_entity.id, OriginalSourceInfoComponent)
    assert osi_comp_retrieved is not None
    assert osi_comp_retrieved.source_type == "referenced_file"

    # Check SHA256 hash (bytes)
    sha256_hex = file_operations.calculate_sha256(sample_image_file)
    sha256_bytes = binascii.unhexlify(sha256_hex)
    sha256_db_comp = ecs_service.get_component(db_session, asset_entity.id, ContentHashSHA256Component)
    assert sha256_db_comp is not None
    assert sha256_db_comp.hash_value == sha256_bytes

    # Check Perceptual pHash (bytes) - assuming image was processed for pHash
    # Note: file_operations.generate_perceptual_hashes_async returns a dict of hex strings
    perceptual_hashes_hex_dict = await file_operations.generate_perceptual_hashes_async(sample_image_file)
    if "phash" in perceptual_hashes_hex_dict:
        phash_hex = perceptual_hashes_hex_dict["phash"]
        phash_bytes = binascii.unhexlify(phash_hex)
        phash_db_comp = ecs_service.get_component(db_session, asset_entity.id, ImagePerceptualPHashComponent)
        assert phash_db_comp is not None
        assert phash_db_comp.hash_value == phash_bytes
    else:
        # If generate_perceptual_hashes_async didn't produce a phash (e.g. non-image file, though fixture is image)
        # then the component might not exist. For this test, we expect it for sample_image_file.
        pytest.fail("pHash was expected for sample_image_file but not generated/found.")


    assert f"Finished AssetReferenceIngestionRequested for Entity ID {asset_entity.id}" in caplog.text


# Tests for query systems

@pytest.mark.asyncio
async def test_handle_find_entity_by_hash_query_system(
    test_world_with_db_session: World, sample_text_file: Path, caplog
):
    """Test the system handling FindEntityByHashQuery events."""
    world = test_world_with_db_session
    db_session = world.get_db_session() # For direct verification if needed, though system should log
    caplog.set_level(logging.INFO)

    from dam.core.events import AssetFileIngestionRequested, FindEntityByHashQuery
    from dam.services import file_operations
    import uuid

    # 1. Ingest a file to ensure it exists
    props = file_operations.get_file_properties(sample_text_file)
    ingest_event = AssetFileIngestionRequested(
        filepath_on_disk=sample_text_file,
        original_filename=props[0],
        mime_type=props[2],
        size_bytes=props[1],
        world_name=world.name
    )
    await world.dispatch_event(ingest_event)

    # Get the ingested entity's ID for verification (indirectly)
    # This is a bit coupled, but necessary to confirm the query found the *correct* entity.
    osi_comps = ecs_service.find_entities_by_component_attribute_value(
        db_session, OriginalSourceInfoComponent, "original_filename", props[0]
    )
    ingested_entity = None
    for entity_candidate in osi_comps:
        osi = ecs_service.get_component(db_session, entity_candidate.id, OriginalSourceInfoComponent)
        if osi and osi.original_path == str(sample_text_file.resolve()):
            ingested_entity = entity_candidate
            break
    assert ingested_entity is not None, "Failed to ingest or find test file for query setup"


    # 2. Dispatch FindEntityByHashQuery event
    sha256_hex = file_operations.calculate_sha256(sample_text_file)
    request_id = str(uuid.uuid4())

    query_event = FindEntityByHashQuery(
        hash_value=sha256_hex,
        hash_type="sha256",
        world_name=world.name,
        request_id=request_id
    )
    caplog.clear() # Clear previous logs from ingestion
    await world.dispatch_event(query_event)

    # 3. Verification: Check logs for the query result
    # The system currently logs the result.
    assert f"System handling FindEntityByHashQuery for hash: {sha256_hex}" in caplog.text
    assert f"[QueryResult RequestID: {request_id}] Found Entity ID: {ingested_entity.id} for hash {sha256_hex}" in caplog.text

    # Test with a non-existent hash
    non_existent_hash_hex = "0000000000000000000000000000000000000000000000000000000000000000"
    request_id_not_found = str(uuid.uuid4())
    query_event_not_found = FindEntityByHashQuery(
        hash_value=non_existent_hash_hex,
        hash_type="sha256",
        world_name=world.name,
        request_id=request_id_not_found
    )
    caplog.clear()
    await world.dispatch_event(query_event_not_found)
    assert f"[QueryResult RequestID: {request_id_not_found}] No entity found for hash {non_existent_hash_hex}" in caplog.text


@pytest.mark.asyncio
async def test_handle_find_similar_images_query_system(
    test_world_with_db_session: World, tmp_path: Path, caplog
):
    """Test the system handling FindSimilarImagesQuery events."""
    world = test_world_with_db_session
    db_session = world.get_db_session()
    caplog.set_level(logging.INFO)

    from dam.core.events import AssetFileIngestionRequested, FindSimilarImagesQuery
    from dam.services import file_operations
    import uuid
    from PIL import Image # To create test images

    # 1. Create and ingest test images
    img_dir = tmp_path / "sim_test_images"
    img_dir.mkdir(exist_ok=True)

    def _create_test_image(name: str, R: int, G: int, B: int) -> Path:
        img_path = img_dir / name
        img = Image.new("RGB", (64, 64), color=(R, G, B))
        img.save(img_path, "PNG")

        props = file_operations.get_file_properties(img_path)
        ingest_event = AssetFileIngestionRequested(
            filepath_on_disk=img_path,
            original_filename=props[0],
            mime_type=props[2],
            size_bytes=props[1],
            world_name=world.name
        )
        asyncio.run(world.dispatch_event(ingest_event)) # Run sync for setup simplicity here
        return img_path

    img1_path = _create_test_image("img1.png", 255, 0, 0) # Red
    img2_path = _create_test_image("img2.png", 250, 5, 5) # Slightly different Red
    _create_test_image("img3.png", 0, 0, 255)   # Blue (different)

    # Get Entity ID for img2 to check if it's found as similar to img1
    img2_props = file_operations.get_file_properties(img2_path)
    osi_img2_entities = ecs_service.find_entities_by_component_attribute_value(
        db_session, OriginalSourceInfoComponent, "original_filename", img2_props[0]
    )
    img2_entity = None
    for entity_candidate in osi_img2_entities:
        osi = ecs_service.get_component(db_session, entity_candidate.id, OriginalSourceInfoComponent)
        if osi and osi.original_path == str(img2_path.resolve()):
            img2_entity = entity_candidate
            break
    assert img2_entity is not None, "Failed to find ingested entity for img2.png"


    # 2. Dispatch FindSimilarImagesQuery event for img1
    request_id = str(uuid.uuid4())
    query_event = FindSimilarImagesQuery(
        image_path=img1_path,
        phash_threshold=8, # Allow some difference
        ahash_threshold=8,
        dhash_threshold=8,
        world_name=world.name,
        request_id=request_id
    )
    caplog.clear()
    await world.dispatch_event(query_event)

    # 3. Verification: Check logs
    assert f"System handling FindSimilarImagesQuery for image: {img1_path.name}" in caplog.text
    # This assertion depends on the exact format of the result logging.
    # A more robust test might involve a way to retrieve structured results if the system supported it.
    # For now, checking if the log mentions the found entity ID:
    assert f"[QueryResult RequestID: {request_id}]" in caplog.text # General result log
    # Example of a more specific check if logs were structured:
    # assert f"'entity_id': {img2_entity.id}" in caplog.text # Check if img2 is mentioned in results
    # This requires knowing how similar_entities_info is logged.
    # A simple check that some results were found:
    assert "Found 1 similar images." in caplog.text or "Found 2 similar images." in caplog.text # Expect img1 itself and img2
    # The current `handle_find_similar_images_query` excludes the source image itself from results.
    # So if img1 is query, and img2 is similar, it should find 1.
    assert f"'entity_id': {img2_entity.id}" in caplog.text # More specific check
