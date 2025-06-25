import json
from pathlib import Path
import pytest
from sqlalchemy.orm import Session

from dam.core import database as app_database # For SessionLocal and test setup
from dam.models import (
    Entity,
    FilePropertiesComponent,
    ContentHashSHA256Component,
    # Import other components as needed for testing specific export/import
)
from dam.services import world_service, asset_service, ecs_service

# Use the same test data dir and image creation helpers if needed, or create new ones.
# For world export/import, we primarily need entities and components in the DB.

@pytest.fixture(scope="function")
def populated_db_session(setup_test_environment_for_world_service) -> Session:
    """
    Provides a test database session populated with some entities and components.
    Relies on a setup fixture similar to test_cli.py's setup_test_environment.
    """
    db = app_database.SessionLocal() # Uses patched SessionLocal from setup

    # Add some assets to create entities and components
    # Using _FIXTURE_IMG_A, _FIXTURE_TXT_FILE from conftest or similar setup if available
    # For simplicity here, let's create some dummy files directly if those globals aren't set up for this module.

    source_files_dir = setup_test_environment_for_world_service # This fixture now returns the source_files_dir
    img_a_path = source_files_dir / "ws_img_A.png"
        # Use source_files_dir, not the fixture 'source_dir' which is a FixtureFunctionDefinition here
        txt_file_path = source_files_dir / "ws_sample.txt"

    if not img_a_path.exists():
        from tests.test_cli import _create_dummy_image # Re-use or adapt
        _create_dummy_image(img_a_path, "red", size=(5,5)) # Smaller for speed
    if not txt_file_path.exists():
        txt_file_path.write_text("world service test content")

    try:
        # Entity 1: Image
        e1, _ = asset_service.add_asset_file(
            session=db,
            filepath_on_disk=img_a_path,
            original_filename="ws_img_A.png",
            mime_type="image/png",
            size_bytes=img_a_path.stat().st_size,
        )

        # Entity 2: Text file
        e2, _ = asset_service.add_asset_file(
            session=db,
            filepath_on_disk=txt_file_path,
            original_filename="ws_sample.txt",
            mime_type="text/plain",
            size_bytes=txt_file_path.stat().st_size,
        )

        # Entity 3: Entity with no FileProperties/Location, but maybe other components later
        e3 = ecs_service.create_entity(db)
        # Example: Add a SHA256 hash component directly to e3 for testing diverse components
        # This is artificial for testing component export diversity.
        sha_comp_e3 = ContentHashSHA256Component(entity_id=e3.id, entity=e3, hash_value="manual_hash_for_e3")
        ecs_service.add_component_to_entity(db, e3.id, sha_comp_e3)

        db.commit()

        # Store IDs for assertions later, as IDs might change if DB is reset per test
        # However, for export/import, we assume IDs are preserved or mapped.
        # For this test, we'll rely on the export containing these.

    except Exception as e:
        db.rollback()
        pytest.fail(f"Failed to populate DB for world service tests: {e}")
    finally:
        # db.close() # Session is yielded, test function will close it.
        pass

    return db


@pytest.fixture(scope="function")
def setup_test_environment_for_world_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    Sets up a minimal test environment (DB, asset storage) for world_service tests.
    Similar to setup_test_environment in test_cli.py but tailored if needed.
    Returns the path to a temporary 'source_files' directory for test assets.
    """
    # This is largely a copy of the setup from test_cli.py, simplified.
    # Consider refactoring into a shared conftest.py if it grows more complex.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from dam.core import database as app_db_module # aliased to avoid conflict
    from dam.core.config import settings as app_settings

    temp_storage_path = tmp_path / "ws_asset_storage"
    temp_storage_path.mkdir(parents=True, exist_ok=True)
    db_file = tmp_path / "test_ws_dam.db"
    test_db_url = f"sqlite:///{db_file}"

    monkeypatch.setenv("DAM_ASSET_STORAGE_PATH", str(temp_storage_path))
    monkeypatch.setenv("DAM_DATABASE_URL", test_db_url)
    monkeypatch.setenv("TESTING_MODE", "True")

    # Reload settings and patch
    app_settings.ASSET_STORAGE_PATH = str(temp_storage_path) # type: ignore
    app_settings.DATABASE_URL = test_db_url # type: ignore
    app_settings.TESTING_MODE = True # type: ignore

    new_engine = create_engine(app_settings.DATABASE_URL, connect_args={"check_same_thread": False})
    monkeypatch.setattr(app_db_module, "engine", new_engine)
    new_session_local = sessionmaker(autocommit=False, autoflush=False, bind=new_engine)
    monkeypatch.setattr(app_db_module, "SessionLocal", new_session_local)

    import dam.models # Ensure models are registered
    app_db_module.create_db_and_tables()

    source_files_dir = tmp_path / "ws_source_files"
    source_files_dir.mkdir(exist_ok=True)

    # Call the component registration function from world_service to ensure types are loaded
    # This is important because tests might run in isolation where app startup doesn't happen.
    world_service._populate_component_types_for_serialization()

    return source_files_dir


def test_export_ecs_world(populated_db_session: Session, tmp_path: Path):
    db = populated_db_session
    export_file = tmp_path / "world_export.json"

    world_service.export_ecs_world_to_json(db, export_file)
    assert export_file.exists()

    with open(export_file, "r") as f:
        data = json.load(f)

    assert "entities" in data
    assert len(data["entities"]) >= 3 # We created at least 3 entities

    # Find data for our specific entities (original filenames are good identifiers for test)
    img_entity_data = next((e for e in data["entities"] if
                            "FilePropertiesComponent" in e["components"] and
                            any(fpc.get("original_filename") == "ws_img_A.png" for fpc in e["components"]["FilePropertiesComponent"])), None)
    txt_entity_data = next((e for e in data["entities"] if
                            "FilePropertiesComponent" in e["components"] and
                            any(fpc.get("original_filename") == "ws_sample.txt" for fpc in e["components"]["FilePropertiesComponent"])), None)
    manual_hash_entity_data = next((e for e in data["entities"] if
                                    "ContentHashSHA256Component" in e["components"] and
                                    any(csha.get("hash_value") == "manual_hash_for_e3" for csha in e["components"]["ContentHashSHA256Component"])), None)

    assert img_entity_data is not None
    assert "FilePropertiesComponent" in img_entity_data["components"]
    assert "ContentHashSHA256Component" in img_entity_data["components"]
    # Potentially ImageDimensionsComponent, perceptual hashes etc. if image processing was complete.

    assert txt_entity_data is not None
    assert "FilePropertiesComponent" in txt_entity_data["components"]

    assert manual_hash_entity_data is not None
    assert "ContentHashSHA256Component" in manual_hash_entity_data["components"]
    assert not manual_hash_entity_data["components"].get("FilePropertiesComponent") # Should not have this one

    # Check component structure (example for FilePropertiesComponent on image)
    fp_comp_data = img_entity_data["components"]["FilePropertiesComponent"][0]
    assert fp_comp_data["__component_type__"] == "FilePropertiesComponent"
    assert fp_comp_data["original_filename"] == "ws_img_A.png"
    assert "entity_id" not in fp_comp_data # Should be excluded by export logic

    db.close()


def test_import_ecs_world_clean_db(setup_test_environment_for_world_service, tmp_path: Path):
    # setup_test_environment_for_world_service creates a clean DB
    db = app_database.SessionLocal() # New session to the clean DB

    source_files_dir = setup_test_environment_for_world_service # to get path for dummy files
    img_a_path = source_files_dir / "ws_img_A.png" # Ensure this matches filename in JSON
    txt_file_path = source_files_dir / "ws_sample.txt"
    if not img_a_path.exists(): # Create if not existing from fixture (e.g. if test run in isolation)
         from tests.test_cli import _create_dummy_image
         _create_dummy_image(img_a_path, "red", size=(5,5))
    if not txt_file_path.exists():
        txt_file_path.write_text("world service test content for import")


    # Create a dummy export file
    export_data = {
        "entities": [
            {
                "id": 101, # Using distinct IDs for test
                "components": {
                    "FilePropertiesComponent": [{
                        "__component_type__": "FilePropertiesComponent",
                        "original_filename": "ws_img_A.png",
                        "file_size_bytes": 100,
                        "mime_type": "image/png"
                    }],
                    "ContentHashSHA256Component": [{
                        "__component_type__": "ContentHashSHA256Component",
                        "hash_value": "fake_sha256_for_import_img"
                    }],
                    # Assume FileLocationComponent would also be here for a real asset.
                    # For --no-copy assets, this path would need to exist if we were fully testing asset_service.
                    # For world import, it just imports component data.
                    "FileLocationComponent": [{
                        "__component_type__": "FileLocationComponent",
                        "file_identifier": str(img_a_path.resolve()), # Dummy path
                        "storage_type": "referenced_local_file", # Or "local_content_addressable"
                        "original_filename": "ws_img_A.png"
                    }]
                }
            },
            {
                "id": 102,
                "components": {
                     "ContentHashSHA256Component": [{
                        "__component_type__": "ContentHashSHA256Component",
                        "hash_value": "fake_sha256_for_import_manual"
                    }]
                }
            }
        ]
    }
    import_file = tmp_path / "world_import_test.json"
    with open(import_file, "w") as f:
        json.dump(export_data, f)

    # Perform import
    # Note: The import logic for Entity IDs is simplified and may have issues with auto-incrementing PKs
    # if not importing into a truly empty table or if IDs are not sequential.
    # For SQLite, setting ID on insert works if table is empty or ID doesn't exist.
    world_service.import_ecs_world_from_json(db, import_file, merge=False)

    # Verify imported data
    # Query by unique data since IDs are auto-assigned and won't match JSON IDs 101, 102.

    # Find entity originally 101 (ws_img_A.png)
    imported_img_entity_props = db.query(FilePropertiesComponent).filter(
        FilePropertiesComponent.original_filename == "ws_img_A.png"
    ).one_or_none()
    assert imported_img_entity_props is not None, "Imported image entity (ws_img_A.png) not found by filename."
    imported_img_entity_id = imported_img_entity_props.entity_id

    fp_comp_img = ecs_service.get_components(db, imported_img_entity_id, FilePropertiesComponent)
    assert len(fp_comp_img) == 1
    assert fp_comp_img[0].original_filename == "ws_img_A.png"
    sha_comp_img = ecs_service.get_components(db, imported_img_entity_id, ContentHashSHA256Component)
    assert len(sha_comp_img) == 1
    assert sha_comp_img[0].hash_value == "fake_sha256_for_import_img"

    # Find entity originally 102 (manual hash)
    imported_manual_hash_entity_sha = db.query(ContentHashSHA256Component).filter(
        ContentHashSHA256Component.hash_value == "fake_sha256_for_import_manual"
    ).one_or_none()
    assert imported_manual_hash_entity_sha is not None, "Imported manual hash entity not found by hash value."
    imported_manual_hash_entity_id = imported_manual_hash_entity_sha.entity_id

    sha_comp_manual = ecs_service.get_components(db, imported_manual_hash_entity_id, ContentHashSHA256Component)
    assert len(sha_comp_manual) == 1
    assert sha_comp_manual[0].hash_value == "fake_sha256_for_import_manual"
    fp_comp_manual = ecs_service.get_components(db, imported_manual_hash_entity_id, FilePropertiesComponent)
    assert len(fp_comp_manual) == 0 # Entity from JSON ID 102 had no FileProperties

    db.close()


def test_import_ecs_world_with_merge(populated_db_session: Session, tmp_path: Path):
    db = populated_db_session # DB already has some data

    # Get an existing entity's ID and details to try to merge/update
    existing_txt_entity_props = db.query(FilePropertiesComponent).filter(
        FilePropertiesComponent.original_filename == "ws_sample.txt"
    ).one()
    existing_txt_entity_id = existing_txt_entity_props.entity_id

    # Create an import file that updates this entity and adds a new one
    updated_txt_filename = "ws_sample_updated.txt"
    new_sha_for_txt = "merged_sha_for_txt"

    export_data_for_merge = {
        "entities": [
            { # Update existing entity (ws_sample.txt)
                "id": existing_txt_entity_id,
                "components": {
                    "FilePropertiesComponent": [{ # This should replace existing FPC
                        "__component_type__": "FilePropertiesComponent",
                        "original_filename": updated_txt_filename, # Changed filename
                        "file_size_bytes": 999, # Changed size
                        "mime_type": "text/merged" # Changed mime
                    }],
                    "ContentHashSHA256Component": [{ # This should replace existing SHA256
                        "__component_type__": "ContentHashSHA256Component",
                        "hash_value": new_sha_for_txt
                    }]
                    # Assume other components for this entity are removed if not in JSON (merge strategy)
                }
            },
            { # Add a new entity
                "id": 201,
                "components": {
                     "ContentHashSHA256Component": [{
                        "__component_type__": "ContentHashSHA256Component",
                        "hash_value": "new_entity_hash_for_merge"
                    }]
                }
            }
        ]
    }
    merge_import_file = tmp_path / "world_merge_import_test.json"
    with open(merge_import_file, "w") as f:
        json.dump(export_data_for_merge, f)

    world_service.import_ecs_world_from_json(db, merge_import_file, merge=True)

    # Verify Entity existing_txt_entity_id
    updated_entity = db.get(Entity, existing_txt_entity_id)
    assert updated_entity is not None

    fp_comps = ecs_service.get_components(db, existing_txt_entity_id, FilePropertiesComponent)
    assert len(fp_comps) == 1 # Should have replaced, not added
    assert fp_comps[0].original_filename == updated_txt_filename
    assert fp_comps[0].file_size_bytes == 999

    sha_comps = ecs_service.get_components(db, existing_txt_entity_id, ContentHashSHA256Component)
    assert len(sha_comps) == 1
    assert sha_comps[0].hash_value == new_sha_for_txt

    # Verify new Entity 201
    new_merged_entity = db.get(Entity, 201)
    assert new_merged_entity is not None
    new_sha_comps = ecs_service.get_components(db, 201, ContentHashSHA256Component)
    assert len(new_sha_comps) == 1
    assert new_sha_comps[0].hash_value == "new_entity_hash_for_merge"

    db.close()

# TODO: Add tests for error conditions in import:
# - File not found
# - Invalid JSON
# - Missing 'entities' key
# - Conflicting entity ID when merge=False (if import logic is changed to raise error)
# - Unknown component type in JSON
# - Failure to create entity with specified ID (if that becomes a hard error)

# Note: `merge_ecs_worlds` and `split_ecs_world` are placeholders and marked
#       with NotImplementedError, so direct tests for them are deferred until implementation.
#       `test_import_ecs_world_with_merge` covers the merge flag of `import_ecs_world_from_json`.

# Need to define source_dir for populated_db_session
# The `monkeypatch` fixture is provided by pytest automatically.
@pytest.fixture
def source_dir(setup_test_environment_for_world_service: Path) -> Path:
    return setup_test_environment_for_world_service
