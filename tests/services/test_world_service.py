import json
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

# Fixtures `db_session` and `settings_override` will be used from conftest.py
from dam.core.database import db_manager
from dam.core.config import settings as app_settings # To get current world name for service calls
from dam.models import (
    Entity,
    FilePropertiesComponent,
    ContentHashSHA256Component,
    FileLocationComponent, # Added for import test
)
from dam.services import world_service, asset_service, ecs_service
from dam.services import file_operations # Import for get_file_properties


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
            from tests.test_cli import _create_dummy_image # Assumes this helper is available
            _create_dummy_image(img_a_path, "red", size=(5,5))
        except ImportError: # Fallback if _create_dummy_image is not found or PIL is missing
            img_a_path.write_text("dummy image content")

    if not txt_file_path.exists():
        txt_file_path.write_text("world service test content for import/export")

    return source_files_dir


@pytest.fixture(scope="function")
def populated_db_session_for_export(db_session: Session, source_files_for_world_service: Path, settings_override) -> Session:
    """
    Provides the default test world session (from db_session fixture) populated
    with some assets for export testing.
    Relies on settings_override being active.
    """
    current_world_name = app_settings.DEFAULT_WORLD_NAME
    if not current_world_name:
        pytest.fail("Default test world name not set in settings_override for populated_db_session_for_export.")

    img_a_path = source_files_for_world_service / "ws_img_A.png"
    txt_file_path = source_files_for_world_service / "ws_sample.txt"

    try:
        img_props = file_operations.get_file_properties(img_a_path)
        asset_service.add_asset_file(
            session=db_session, filepath_on_disk=img_a_path, original_filename="ws_img_A.png",
            mime_type=img_props[2], size_bytes=img_props[1], world_name=current_world_name
        )
        txt_props = file_operations.get_file_properties(txt_file_path)
        asset_service.add_asset_file(
            session=db_session, filepath_on_disk=txt_file_path, original_filename="ws_sample.txt",
            mime_type=txt_props[2], size_bytes=txt_props[1], world_name=current_world_name
        )
        e3 = ecs_service.create_entity(db_session)
        sha_comp_e3 = ContentHashSHA256Component(entity_id=e3.id, entity=e3, hash_value="manual_hash_for_e3")
        ecs_service.add_component_to_entity(db_session, e3.id, sha_comp_e3)
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        pytest.fail(f"Failed to populate DB for world service export tests: {e}")

    return db_session


def test_export_ecs_world(populated_db_session_for_export: Session, tmp_path: Path, settings_override):
    db_session = populated_db_session_for_export
    current_world_name = app_settings.DEFAULT_WORLD_NAME
    export_file = tmp_path / "world_export.json"

    world_service.export_ecs_world_to_json(db_session, export_file, world_name_for_log=current_world_name)
    assert export_file.exists()

    with open(export_file, "r") as f:
        data = json.load(f)

    assert "entities" in data
    assert len(data["entities"]) == 3

    img_entity_data = next((e for e in data["entities"] if "FilePropertiesComponent" in e["components"] and any(fpc.get("original_filename") == "ws_img_A.png" for fpc in e["components"]["FilePropertiesComponent"])), None)
    txt_entity_data = next((e for e in data["entities"] if "FilePropertiesComponent" in e["components"] and any(fpc.get("original_filename") == "ws_sample.txt" for fpc in e["components"]["FilePropertiesComponent"])), None)
    manual_hash_entity_data = next((e for e in data["entities"] if "ContentHashSHA256Component" in e["components"] and any(csha.get("hash_value") == "manual_hash_for_e3" for csha in e["components"]["ContentHashSHA256Component"])), None)

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
def clean_db_session_for_import(another_db_session: Session, settings_override) -> Session:
    # settings_override ensures the app_settings.DEFAULT_WORLD_NAME is patched if needed,
    # and test_db_manager (used by another_db_session) is correctly initialized.
    # another_db_session is for "test_world_beta"
    return another_db_session

def test_import_ecs_world_clean_db(clean_db_session_for_import: Session, source_files_for_world_service: Path, tmp_path: Path):
    db_session = clean_db_session_for_import
    # The world name for another_db_session is "test_world_beta"
    current_world_name_for_log = "test_world_beta"

    img_a_path = source_files_for_world_service / "ws_img_A.png"

    export_data = {
        "entities": [
            {"id": 101, "components": {
                "FilePropertiesComponent": [{"__component_type__": "FilePropertiesComponent", "original_filename": "ws_img_A.png", "file_size_bytes": 100, "mime_type": "image/png"}],
                "ContentHashSHA256Component": [{"__component_type__": "ContentHashSHA256Component", "hash_value": "fake_sha256_for_import_img"}],
                "FileLocationComponent": [{"__component_type__": "FileLocationComponent", "file_identifier": str(img_a_path.resolve()), "storage_type": "referenced_local_file", "original_filename": "ws_img_A.png"}]
            }},
            {"id": 102, "components": {
                 "ContentHashSHA256Component": [{"__component_type__": "ContentHashSHA256Component", "hash_value": "fake_sha256_for_import_manual"}]
            }}
        ]
    }
    import_file = tmp_path / "world_import_test.json"
    with open(import_file, "w") as f: json.dump(export_data, f)

    world_service.import_ecs_world_from_json(db_session, import_file, merge=False, world_name_for_log=current_world_name_for_log)

    imported_img_props = db_session.query(FilePropertiesComponent).filter(FilePropertiesComponent.original_filename == "ws_img_A.png").one_or_none()
    assert imported_img_props is not None
    img_entity_id = imported_img_props.entity_id
    assert ecs_service.get_component(db_session, img_entity_id, ContentHashSHA256Component).hash_value == "fake_sha256_for_import_img"

    manual_hash_comp = db_session.query(ContentHashSHA256Component).filter(ContentHashSHA256Component.hash_value == "fake_sha256_for_import_manual").one_or_none()
    assert manual_hash_comp is not None
    assert not ecs_service.get_component(db_session, manual_hash_comp.entity_id, FilePropertiesComponent)


def test_import_ecs_world_with_merge(populated_db_session_for_export: Session, tmp_path: Path, settings_override):
    db_session = populated_db_session_for_export
    current_world_name_for_log = app_settings.DEFAULT_WORLD_NAME # Should be "test_world_alpha"

    existing_txt_entity_props = db_session.query(FilePropertiesComponent).filter(FilePropertiesComponent.original_filename == "ws_sample.txt").one()
    existing_txt_entity_id = existing_txt_entity_props.entity_id

    updated_txt_filename = "ws_sample_updated.txt"
    new_sha_for_txt = "merged_sha_for_txt"
    new_entity_json_id = 201

    export_data_for_merge = {
        "entities": [
            {"id": existing_txt_entity_id, "components": {
                "FilePropertiesComponent": [{"__component_type__": "FilePropertiesComponent", "original_filename": updated_txt_filename, "file_size_bytes": 999, "mime_type": "text/merged"}],
                "ContentHashSHA256Component": [{"__component_type__": "ContentHashSHA256Component", "hash_value": new_sha_for_txt}]
            }},
            {"id": new_entity_json_id, "components": {
                 "ContentHashSHA256Component": [{"__component_type__": "ContentHashSHA256Component", "hash_value": "new_entity_hash_for_merge"}]
            }}
        ]
    }
    merge_import_file = tmp_path / "world_merge_import_test.json"
    with open(merge_import_file, "w") as f: json.dump(export_data_for_merge, f)

    world_service.import_ecs_world_from_json(db_session, merge_import_file, merge=True, world_name_for_log=current_world_name_for_log)

    updated_fpc = ecs_service.get_component(db_session, existing_txt_entity_id, FilePropertiesComponent)
    assert updated_fpc.original_filename == updated_txt_filename and updated_fpc.file_size_bytes == 999
    updated_sha = ecs_service.get_component(db_session, existing_txt_entity_id, ContentHashSHA256Component)
    assert updated_sha.hash_value == new_sha_for_txt

    newly_added_entity_sha_comp = db_session.query(ContentHashSHA256Component).filter(ContentHashSHA256Component.hash_value == "new_entity_hash_for_merge").one_or_none()
    assert newly_added_entity_sha_comp is not None
    assert newly_added_entity_sha_comp.entity_id != new_entity_json_id
    assert db_session.get(Entity, newly_added_entity_sha_comp.entity_id) is not None

# The old source_dir fixture and setup_test_environment_for_world_service are removed.
# All tests now rely on conftest.py fixtures (settings_override, db_session, another_db_session, test_db_manager)
# and the local source_files_for_world_service fixture for creating test files.
