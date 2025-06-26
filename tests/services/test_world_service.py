import json
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

# Fixtures `db_session` and `settings_override` will be used from conftest.py
from dam.core.stages import SystemStage  # Added import
from dam.core.system_params import WorldContext  # Added import
from dam.models import (
    ContentHashSHA256Component,
    Entity,
    FilePropertiesComponent,
)
from dam.services import (
    asset_service,
    ecs_service,
    file_operations,  # Import for get_file_properties
    world_service,
)


@pytest.fixture(scope="function")
def source_files_for_world_service(tmp_path: Path) -> Path:
    """
    Creates a temporary 'source_files' directory and populates it with
    dummy files for world_service tests.
    """
    source_files_dir = tmp_path / "ws_source_files"
    source_files_dir.mkdir(exist_ok=True)

    img_a_path = source_files_dir / "ws_img_A.png"
    txt_file_path = source_files_dir / "ws_sample.txt"

    if not img_a_path.exists():
        try:
            from tests.test_cli import _create_dummy_image  # Assumes this helper is available

            _create_dummy_image(img_a_path, "red", size=(5, 5))
        except ImportError:  # Fallback if _create_dummy_image is not found or PIL is missing
            img_a_path.write_text("dummy image content")

    if not txt_file_path.exists():
        txt_file_path.write_text("world service test content for import/export")

    return source_files_dir


# Import World for type hinting
from dam.core.world import World


@pytest.fixture(scope="function")
async def populated_world_for_export(
    test_world_alpha: World, source_files_for_world_service: Path
) -> World: # Made async, depends on test_world_alpha
    """
    Provides the 'test_world_alpha' World instance populated with some assets for export testing.
    """
    world = test_world_alpha
    # settings_override is implicitly handled by test_world_alpha fixture

    img_a_path = source_files_for_world_service / "ws_img_A.png"
    txt_file_path = source_files_for_world_service / "ws_sample.txt"

    # Get a session from the world for populating data
    session = world.get_db_session()
    try:
        img_props = file_operations.get_file_properties(img_a_path)
        # asset_service.add_asset_file now takes world_config
        asset_service.add_asset_file(
            session=session,
            filepath_on_disk=img_a_path,
            original_filename="ws_img_A.png",
            mime_type=img_props[2],
            size_bytes=img_props[1],
            world_config=world.config, # Pass world's config
        )
        txt_props = file_operations.get_file_properties(txt_file_path)
        asset_service.add_asset_file(
            session=session,
            filepath_on_disk=txt_file_path,
            original_filename="ws_sample.txt",
            mime_type=txt_props[2],
            size_bytes=txt_props[1],
            world_config=world.config, # Pass world's config
        )
        e3 = ecs_service.create_entity(session)
        sha_comp_e3 = ContentHashSHA256Component(entity_id=e3.id, entity=e3, hash_value="manual_hash_for_e3")
        ecs_service.add_component_to_entity(session, e3.id, sha_comp_e3)
        session.commit()
    except Exception as e:
        session.rollback()
        pytest.fail(f"Failed to populate DB for world '{world.name}' for export tests: {e}")
    finally:
        session.close()

    # Run METADATA_EXTRACTION stage using the world's method
    # This will use its own session management internally.
    await world.execute_stage(SystemStage.METADATA_EXTRACTION)

    return world


@pytest.mark.asyncio
async def test_export_ecs_world(populated_world_for_export: World, tmp_path: Path):
    # populated_world_for_export is now a World instance
    # settings_override is implicitly handled by the World fixture

    export_world = await populated_world_for_export # Fixture is async
    export_file = tmp_path / "world_export.json"

    # world_service.export_ecs_world_to_json now takes a World object
    world_service.export_ecs_world_to_json(export_world, export_file)
    assert export_file.exists()

    with open(export_file, "r") as f:
        data = json.load(f)

    assert "entities" in data
    assert len(data["entities"]) == 3

    img_entity_data = next(
        (
            e
            for e in data["entities"]
            if "FilePropertiesComponent" in e["components"]
            and any(
                fpc.get("original_filename") == "ws_img_A.png" for fpc in e["components"]["FilePropertiesComponent"]
            )
        ),
        None,
    )
    txt_entity_data = next(
        (
            e
            for e in data["entities"]
            if "FilePropertiesComponent" in e["components"]
            and any(
                fpc.get("original_filename") == "ws_sample.txt" for fpc in e["components"]["FilePropertiesComponent"]
            )
        ),
        None,
    )
    manual_hash_entity_data = next(
        (
            e
            for e in data["entities"]
            if "ContentHashSHA256Component" in e["components"]
            and any(
                csha.get("hash_value") == "manual_hash_for_e3" for csha in e["components"]["ContentHashSHA256Component"]
            )
        ),
        None,
    )

    assert img_entity_data is not None
    assert "FilePropertiesComponent" in img_entity_data["components"]
    assert "ContentHashSHA256Component" in img_entity_data["components"]
    assert txt_entity_data is not None
    assert "FilePropertiesComponent" in txt_entity_data["components"]
    assert manual_hash_entity_data is not None
    assert "ContentHashSHA256Component" in manual_hash_entity_data["components"]
    assert "FilePropertiesComponent" not in manual_hash_entity_data["components"]

    fp_comp_data = img_entity_data["components"]["FilePropertiesComponent"][0]
    assert fp_comp_data["__component_type__"] == "FilePropertiesComponent"
    assert fp_comp_data["original_filename"] == "ws_img_A.png"
    assert "entity_id" not in fp_comp_data


@pytest.fixture
def clean_world_for_import(test_world_beta: World) -> World:
    # test_world_beta fixture (from conftest) already ensures settings_override is active
    # and the world is set up with a clean DB.
    return test_world_beta


def test_import_ecs_world_clean_db(
    clean_world_for_import: World, source_files_for_world_service: Path, tmp_path: Path
):
    import_world = clean_world_for_import # World instance to import into
    # current_world_name_for_log = import_world.name # For logging if needed, but service handles it

    img_a_path = source_files_for_world_service / "ws_img_A.png"

    # Note: The FileLocationComponent in export_data uses "file_identifier" and "storage_type": "referenced_local_file".
    # The import logic in world_service.py has some mapping for FileLocationComponent fields like:
    # - "file_identifier" -> "content_identifier"
    # - "original_filename" -> "contextual_filename"
    # - "filepath" -> "physical_path_or_key"
    # Ensure this JSON matches what the import expects or update the JSON/import logic.
    # Current FileLocationComponent model expects: content_identifier, physical_path_or_key, contextual_filename, storage_type.
    # The JSON uses 'file_identifier' which should map to 'content_identifier'.
    # It uses 'original_filename' which should map to 'contextual_filename'.
    # It's missing 'physical_path_or_key'. For 'referenced_local_file', this should be the actual path.
    # Let's adjust the JSON to be more aligned or ensure mapping handles it.
    # The service currently warns if physical_path_or_key is missing.
    # For "referenced_local_file", the physical_path_or_key IS the file_identifier (the original path).
    # Let's make the JSON reflect the new model expectations for clarity or test the mapping.
    # The service has: if "file_identifier" in comp_data_cleaned and "content_identifier" not in comp_data_cleaned:
    # comp_data_cleaned["content_identifier"] = comp_data_cleaned.pop("file_identifier")
    # This is okay.
    # Missing physical_path_or_key will be an issue if not mapped from somewhere.
    # Let's assume for "referenced_local_file", the 'file_identifier' (which becomes content_identifier)
    # is NOT the physical path. The physical path is the original path.
    # The old `FileLocationComponent` had `filepath` which was the physical path.
    # The new one has `physical_path_or_key`.
    # The import code has:
    # if ("filepath" in comp_data_cleaned and "physical_path_or_key" not in comp_data_cleaned):
    #    comp_data_cleaned["physical_path_or_key"] = comp_data_cleaned.pop("filepath")
    # The JSON below doesn't have 'filepath'. It has 'file_identifier' for the path.
    # This means the import logic needs to be robust for old JSON formats or the test JSON needs update.
    # Let's assume the import should handle the old format where `file_identifier` for a reference was the path.
    # The current import logic might map file_identifier to content_identifier, then physical_path_or_key is missing.
    # This needs to be fixed in the import logic or the test data.
    # For now, I'll provide physical_path_or_key directly in the test JSON matching the new model.

    export_data = {
        "entities": [
            {
                "id": 101, # This ID might not be preserved if DB auto-increments
                "components": {
                    "FilePropertiesComponent": [
                        {"__component_type__": "FilePropertiesComponent", "original_filename": "ws_img_A.png", "file_size_bytes": 100, "mime_type": "image/png"}
                    ],
                    "ContentHashSHA256Component": [
                        {"__component_type__": "ContentHashSHA256Component", "hash_value": "fake_sha256_for_import_img"}
                    ],
                    "FileLocationComponent": [ # Updated to reflect new model fields more directly
                        {
                            "__component_type__": "FileLocationComponent",
                            "content_identifier": "fake_sha256_for_import_img", # Typically the hash
                            "physical_path_or_key": str(img_a_path.resolve()), # Actual path for reference
                            "contextual_filename": "ws_img_A.png", # Original filename for this location
                            "storage_type": "local_reference", # Changed from referenced_local_file
                        }
                    ],
                },
            },
            {
                "id": 102,
                "components": {
                    "ContentHashSHA256Component": [
                        {"__component_type__": "ContentHashSHA256Component", "hash_value": "fake_sha256_for_import_manual"}
                    ]
                },
            },
        ]
    }
    import_file = tmp_path / "world_import_test.json"
    with open(import_file, "w") as f:
        json.dump(export_data, f)

    # world_service.import_ecs_world_from_json now takes a World object
    world_service.import_ecs_world_from_json(import_world, import_file, merge=False)

    # Use a new session from the import_world for checks
    session = import_world.get_db_session()
    try:
        imported_img_props = (
            session.query(FilePropertiesComponent)
            .filter(FilePropertiesComponent.original_filename == "ws_img_A.png")
            .one_or_none()
        )
        assert imported_img_props is not None
        img_entity_id = imported_img_props.entity_id # This ID is generated by the DB, not from JSON's id:101

        retrieved_sha256_comp = ecs_service.get_component(session, img_entity_id, ContentHashSHA256Component)
        assert retrieved_sha256_comp is not None
        assert retrieved_sha256_comp.hash_value == "fake_sha256_for_import_img"

        manual_hash_comp = (
            session.query(ContentHashSHA256Component)
            .filter(ContentHashSHA256Component.hash_value == "fake_sha256_for_import_manual")
            .one_or_none()
        )
        assert manual_hash_comp is not None
        # Ensure the entity ID for manual_hash_comp is different from img_entity_id if they were separate in JSON
        assert manual_hash_comp.entity_id != img_entity_id
        assert not ecs_service.get_component(session, manual_hash_comp.entity_id, FilePropertiesComponent)
    finally:
        session.close()


@pytest.mark.asyncio
async def test_import_ecs_world_with_merge(populated_world_for_export: World, tmp_path: Path):
    # populated_world_for_export is already a World instance
    # settings_override is implicitly handled
    world_to_merge_into = await populated_world_for_export # Use the same populated world

    # Get a session to query initial state
    session = world_to_merge_into.get_db_session()
    try:
        existing_txt_entity_props = (
            session.query(FilePropertiesComponent)
        .filter(FilePropertiesComponent.original_filename == "ws_sample.txt")
        .one()
    )
    existing_txt_entity_id = existing_txt_entity_props.entity_id

    updated_txt_filename = "ws_sample_updated.txt"
    new_sha_for_txt = "merged_sha_for_txt"
    new_entity_json_id = 201

        existing_txt_entity_props = (
            session.query(FilePropertiesComponent)
            .filter(FilePropertiesComponent.original_filename == "ws_sample.txt")
            .one()
        )
        existing_txt_entity_id = existing_txt_entity_props.entity_id
    finally:
        session.close()

    updated_txt_filename = "ws_sample_updated.txt"
    new_sha_for_txt = "merged_sha_for_txt"
    new_entity_json_id = 201 # Original ID from JSON, will likely change on import

    export_data_for_merge = {
        "entities": [
            {
                "id": existing_txt_entity_id, # Try to merge with this existing entity
                "components": {
                    "FilePropertiesComponent": [
                        {"__component_type__": "FilePropertiesComponent", "original_filename": updated_txt_filename, "file_size_bytes": 999, "mime_type": "text/merged"}
                    ],
                    "ContentHashSHA256Component": [
                        {"__component_type__": "ContentHashSHA256Component", "hash_value": new_sha_for_txt}
                    ],
                },
            },
            {
                "id": new_entity_json_id, # This should be a new entity
                "components": {
                    "ContentHashSHA256Component": [
                        {"__component_type__": "ContentHashSHA256Component", "hash_value": "new_entity_hash_for_merge"}
                    ]
                },
            },
        ]
    }
    merge_import_file = tmp_path / "world_merge_import_test.json"
    with open(merge_import_file, "w") as f:
        json.dump(export_data_for_merge, f)

    world_service.import_ecs_world_from_json(world_to_merge_into, merge_import_file, merge=True)

    # Use a new session for checks after import
    session_after_merge = world_to_merge_into.get_db_session()
    try:
        updated_fpc = ecs_service.get_component(session_after_merge, existing_txt_entity_id, FilePropertiesComponent)
        assert updated_fpc is not None, "FilePropertiesComponent should still exist for the merged entity"
        assert updated_fpc.original_filename == updated_txt_filename
        assert updated_fpc.file_size_bytes == 999

        updated_sha = ecs_service.get_component(session_after_merge, existing_txt_entity_id, ContentHashSHA256Component)
        assert updated_sha is not None, "ContentHashSHA256Component should exist for the merged entity"
        assert updated_sha.hash_value == new_sha_for_txt

        newly_added_entity_sha_comp = (
            session_after_merge.query(ContentHashSHA256Component)
            .filter(ContentHashSHA256Component.hash_value == "new_entity_hash_for_merge")
            .one_or_none()
        )
        assert newly_added_entity_sha_comp is not None, "New entity from merge should exist"
        # The ID of the newly added entity will be DB-generated, not new_entity_json_id from the file.
        assert newly_added_entity_sha_comp.entity_id != new_entity_json_id
        assert session_after_merge.get(Entity, newly_added_entity_sha_comp.entity_id) is not None
    finally:
        session_after_merge.close()


# The old source_dir fixture and setup_test_environment_for_world_service are removed.
# All tests now rely on conftest.py fixtures (settings_override, db_session, another_db_session, test_db_manager)
# and the local source_files_for_world_service fixture for creating test files.
