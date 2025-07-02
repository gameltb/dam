import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import gradio as gr # Import Gradio for type hinting and gr.Dropdown checks

# Functions/classes to be tested
from dam.gradio_ui import (
    set_active_world_and_refresh_dropdowns,
    list_assets_gr,
    get_asset_details_gr,
    add_assets_ui,
    find_by_hash_ui,
    # ... other functions from gradio_ui that will be tested ...
    export_world_ui, # Example world op
)

# Mocked dependencies
from dam.core.world import World
from dam.services import ecs_service, world_service, file_operations
from dam.core.events import FindEntityByHashQuery # For find_by_hash_ui test

# Pytest marker for async tests
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_world():
    """Fixture to create a mock World object."""
    world = MagicMock(spec=World)
    world.name = "test_world"

    # Mock the async session context manager
    mock_session = AsyncMock() # This will be the session object

    # Configure get_db_session to return an async context manager
    # that yields mock_session
    async def async_context_manager(*args, **kwargs):
        return mock_session

    world.get_db_session = MagicMock(return_value=MagicMock(__aenter__=async_context_manager, __aexit__=AsyncMock(return_value=False)))

    # Mock dispatch_event and execute_stage as AsyncMock if they are awaitable
    world.dispatch_event = AsyncMock()
    world.execute_stage = AsyncMock()

    return world


@pytest.fixture(autouse=True)
def mock_active_world(monkeypatch, mock_world):
    """
    Fixture to mock the global _active_world in dam.gradio_ui.
    It's autouse=True so it applies to all tests in this file.
    It also takes mock_world fixture to set _active_world to a controlled mock.
    """
    monkeypatch.setattr("dam.gradio_ui._active_world", mock_world)
    return mock_world # Return it in case a test needs direct access to the patched mock


@pytest.fixture
def mock_get_current_world(monkeypatch, mock_world):
    """Mocks the get_current_world utility function."""
    # This mock will return our standard mock_world if "test_world" is passed,
    # or None otherwise, simulating how get_current_world might behave.
    def _get_current_world_mock(world_name: Optional[str]):
        if world_name == "test_world":
            return mock_world
        return None

    mock_func = MagicMock(side_effect=_get_current_world_mock)
    monkeypatch.setattr("dam.gradio_ui.get_current_world", mock_func)
    return mock_func

async def test_set_active_world_valid_world(mock_active_world, mock_get_current_world):
    """Test set_active_world_and_refresh_dropdowns with a valid world name."""

    # Configure the mock session to return some MIME types
    mock_session = mock_active_world.get_db_session.return_value.__aenter__.return_value
    mock_execute_result = MagicMock()
    mock_execute_result.all.return_value = [("image/jpeg",), ("application/pdf",)]
    mock_session.execute = AsyncMock(return_value=mock_execute_result)

    # Patch the choice-loading functions for other dropdowns to return simple defaults
    with patch("dam.gradio_ui.get_transcode_profile_choices", new_callable=AsyncMock) as mock_get_tp, \
         patch("dam.gradio_ui.get_evaluation_run_choices", new_callable=AsyncMock) as mock_get_er:
        mock_get_tp.return_value = ["default_profile"]
        mock_get_er.return_value = ["default_run"]

        status, mime_dd, tp_dd, er_exec_dd, er_report_dd = await set_active_world_and_refresh_dropdowns("test_world")

    assert "Success: World 'test_world' selected." in status
    assert "MIME types loaded." in status

    # Check mime_type_filter_input (gr.Dropdown)
    assert isinstance(mime_dd, gr.Dropdown)
    assert "" in mime_dd.choices
    assert "image/jpeg" in mime_dd.choices
    assert "application/pdf" in mime_dd.choices
    assert mime_dd.value == ""

    # Check other dropdowns
    assert isinstance(tp_dd, gr.Dropdown)
    assert tp_dd.choices == ["default_profile"]
    assert isinstance(er_exec_dd, gr.Dropdown)
    assert er_exec_dd.choices == ["default_run"]
    assert isinstance(er_report_dd, gr.Dropdown)
    assert er_report_dd.choices == ["default_run"]

    mock_get_current_world.assert_called_once_with("test_world")
    mock_active_world.get_db_session.assert_called_once() # Check session was used for MIME types


async def test_set_active_world_invalid_world(mock_get_current_world):
    """Test set_active_world_and_refresh_dropdowns with an invalid world name."""
    mock_get_current_world.return_value = None # Ensure get_current_world returns None

    status, mime_dd, tp_dd, er_exec_dd, er_report_dd = await set_active_world_and_refresh_dropdowns("invalid_world")

    assert status == "Error: Failed to select world or no valid world chosen."
    assert mime_dd.choices == [""] # Default empty choice
    assert tp_dd.choices == ["Refresh to load..."]
    assert er_exec_dd.choices == ["Refresh to load..."]
    assert er_report_dd.choices == ["Refresh to load..."]
    mock_get_current_world.assert_called_once_with("invalid_world")


async def test_set_active_world_mime_type_load_failure(mock_active_world, mock_get_current_world):
    """Test set_active_world when MIME type loading fails."""
    mock_session = mock_active_world.get_db_session.return_value.__aenter__.return_value
    mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

    with patch("dam.gradio_ui.get_transcode_profile_choices", new_callable=AsyncMock) as mock_get_tp, \
         patch("dam.gradio_ui.get_evaluation_run_choices", new_callable=AsyncMock) as mock_get_er:
        mock_get_tp.return_value = ["default_profile"] # Assume these still load
        mock_get_er.return_value = ["default_run"]

        status, mime_dd, _, _, _ = await set_active_world_and_refresh_dropdowns("test_world")

    assert "Success: World 'test_world' selected." in status
    assert "Error loading MIME types: DB error" in status
    assert mime_dd.choices == ["", "Error: Could not load MIME types"]


async def test_list_assets_gr_no_active_world():
    """Test list_assets_gr when no world is active."""
    with patch("dam.gradio_ui._active_world", None): # Temporarily set _active_world to None for this test
        df_update, status = await list_assets_gr(filename_filter="", mime_type_filter="")

    assert status == "Error: No active world selected. Please select a world first."
    assert df_update.value is None


async def test_list_assets_gr_success_no_filters(mock_active_world):
    """Test list_assets_gr successfully retrieves assets with no filters."""
    mock_session = mock_active_world.get_db_session.return_value.__aenter__.return_value

    # Mock for count query
    mock_count_result = MagicMock()
    mock_count_result.scalar_one_or_none.return_value = 2

    # Mock for data query
    mock_data_result = MagicMock()
    mock_data_result.all.return_value = [
        (1, "file1.jpg", "image/jpeg"),
        (2, "file2.png", "image/png"),
    ]
    # session.execute needs to handle two calls: one for count, one for data
    mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_data_result])

    df_update, status = await list_assets_gr(filename_filter="", mime_type_filter="")

    assert "Displaying 2 of 2 assets (Page 1)." in status
    assert isinstance(df_update, gr.DataFrame)
    assert df_update.value["headers"] == ["ID", "Filename", "MIME Type"]
    assert df_update.value["data"] == [
        (1, "file1.jpg", "image/jpeg"),
        (2, "file2.png", "image/png"),
    ]
    # Check that execute was called twice (for count and for data)
    assert mock_session.execute.call_count == 2


async def test_list_assets_gr_with_filters(mock_active_world):
    """Test list_assets_gr with filename and MIME type filters."""
    mock_session = mock_active_world.get_db_session.return_value.__aenter__.return_value

    mock_count_result = MagicMock()
    mock_count_result.scalar_one_or_none.return_value = 1
    mock_data_result = MagicMock()
    mock_data_result.all.return_value = [(1, "test_file.jpg", "image/jpeg")]
    mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_data_result])

    df_update, status = await list_assets_gr(filename_filter="test_file", mime_type_filter="image/jpeg")

    assert "Displaying 1 of 1 assets (Page 1)." in status
    assert df_update.value["data"] == [(1, "test_file.jpg", "image/jpeg")]

    # Check that the SQL queries would have contained the filters (simplified check)
    # This requires inspecting the mock_session.execute.call_args or call_args_list
    # For simplicity, we trust the function's internal logic given the mocked return values.
    # A more detailed test would assert the structure of the SQLAlchemy select objects passed to execute.
    assert mock_session.execute.call_count == 2


async def test_list_assets_gr_pagination(mock_active_world):
    """Test list_assets_gr pagination logic."""
    mock_session = mock_active_world.get_db_session.return_value.__aenter__.return_value

    mock_count_result = MagicMock()
    mock_count_result.scalar_one_or_none.return_value = 25 # Total assets
    mock_data_result = MagicMock()
    # Assume page_size is default 20, so page 2 should show 5 assets
    mock_data_result.all.return_value = [(i, f"file{i}.txt", "text/plain") for i in range(21, 26)]
    mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_data_result])

    df_update, status = await list_assets_gr(filename_filter="", mime_type_filter="", current_page=2)

    assert "Displaying 5 of 25 assets (Page 2)." in status
    assert len(df_update.value["data"]) == 5
    assert df_update.value["data"][0][0] == 21 # Check first item of page 2
    assert mock_session.execute.call_count == 2
    # We can also inspect the call_args for the select object to check offset and limit
    # data_query_call_args = mock_session.execute.call_args_list[1] # Second call is data query
    # select_statement = data_query_call_args[0][0] # The select object
    # assert select_statement._offset_clause.value == 20 # (2-1) * 20
    # assert select_statement._limit_clause.value == 20
    # This level of detail depends on how SQLAlchemy objects are structured and might be brittle.


async def test_list_assets_gr_no_results(mock_active_world):
    """Test list_assets_gr when no assets match filters or exist."""
    mock_session = mock_active_world.get_db_session.return_value.__aenter__.return_value
    mock_count_result = MagicMock()
    mock_count_result.scalar_one_or_none.return_value = 0
    mock_data_result = MagicMock()
    mock_data_result.all.return_value = []
    mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_data_result])

    df_update, status = await list_assets_gr(filename_filter="nonexistent", mime_type_filter="")

    assert "Info: No assets found matching criteria." in status
    assert len(df_update.value["data"]) == 0


async def test_list_assets_gr_db_error(mock_active_world):
    """Test list_assets_gr when a database error occurs."""
    mock_session = mock_active_world.get_db_session.return_value.__aenter__.return_value
    mock_session.execute = AsyncMock(side_effect=Exception("Database connection failed"))

    df_update, status = await list_assets_gr(filename_filter="", mime_type_filter="")

    assert "Error: Could not load assets. Details: Database connection failed" in status
    assert df_update.value is None


async def test_get_asset_details_gr_no_active_world():
    """Test get_asset_details_gr when no world is active."""
    mock_select_event = MagicMock(spec=gr.SelectData)
    mock_select_event.value = 1
    mock_select_event.index = (0,0) # Simulate selection of first cell

    with patch("dam.gradio_ui._active_world", None):
        json_update = await get_asset_details_gr(mock_select_event)

    assert json_update.value == {"error": "Error: No active world selected."}


async def test_get_asset_details_gr_invalid_selection_event():
    """Test get_asset_details_gr with an invalid selection event."""
    mock_select_event = MagicMock(spec=gr.SelectData)
    mock_select_event.value = None # Simulate no value from click
    mock_select_event.index = (None, None) # Simulate invalid index

    json_update = await get_asset_details_gr(mock_select_event)
    assert json_update.value == {"info": "Info: Click on an Asset ID in the table above to view its details."}


async def test_get_asset_details_gr_asset_not_found(mock_active_world):
    """Test get_asset_details_gr when the asset ID is not found."""
    mock_select_event = MagicMock(spec=gr.SelectData)
    mock_select_event.value = 999 # A non-existent asset ID
    mock_select_event.index = (0,0)

    mock_session = mock_active_world.get_db_session.return_value.__aenter__.return_value
    # Mock ecs_service.get_entity to return None, simulating asset not found
    with patch("dam.gradio_ui.ecs_service.get_entity", AsyncMock(return_value=None)) as mock_get_entity:
        json_update = await get_asset_details_gr(mock_select_event)

    mock_get_entity.assert_called_once_with(mock_session, 999)
    assert json_update.value == {"error": "Error: Asset ID 999 not found."}


async def test_get_asset_details_gr_success(mock_active_world):
    """Test get_asset_details_gr successfully retrieves component data."""
    mock_select_event = MagicMock(spec=gr.SelectData)
    mock_select_event.value = 1 # Assume asset ID 1 is selected
    mock_select_event.index = (0,0)

    mock_session = mock_active_world.get_db_session.return_value.__aenter__.return_value

    # Mock entity return
    mock_entity = MagicMock(spec=Entity)
    mock_entity.id = 1

    # Mock components
    # We need to mock the __table__.columns structure for serialization
    mock_fpc_instance = MagicMock()
    mock_fpc_instance.original_filename = "test.jpg"
    mock_fpc_instance.mime_type = "image/jpeg"
    # Create mock columns for FilePropertiesComponent
    fpc_col_fn = MagicMock(); fpc_col_fn.key = "original_filename"
    fpc_col_mime = MagicMock(); fpc_col_mime.key = "mime_type"
    mock_fpc_instance.__table__ = MagicMock()
    mock_fpc_instance.__table__.columns = [fpc_col_fn, fpc_col_mime]


    mock_idc_instance = MagicMock()
    mock_idc_instance.width = 100
    mock_idc_instance.height = 200
    idc_col_w = MagicMock(); idc_col_w.key = "width"
    idc_col_h = MagicMock(); idc_col_h.key = "height"
    mock_idc_instance.__table__ = MagicMock()
    mock_idc_instance.__table__.columns = [idc_col_w, idc_col_h]


    # Patch REGISTERED_COMPONENT_TYPES used by the function
    # And ecs_service.get_components and ecs_service.get_entity
    # Need to import the actual component classes for spec if patching get_components
    from dam.models.properties import FilePropertiesComponent, ImageDimensionsComponent

    with patch("dam.gradio_ui.ecs_service.get_entity", AsyncMock(return_value=mock_entity)) as mock_get_entity, \
         patch("dam.gradio_ui.ecs_service.get_components", new_callable=AsyncMock) as mock_get_components, \
         patch("dam.gradio_ui.REGISTERED_COMPONENT_TYPES", [FilePropertiesComponent, ImageDimensionsComponent]): # Patch the list directly

        # Define side effect for get_components based on component type
        def get_components_side_effect(session, entity_id, comp_type_cls):
            if comp_type_cls == FilePropertiesComponent:
                return [mock_fpc_instance]
            elif comp_type_cls == ImageDimensionsComponent:
                return [mock_idc_instance]
            return []
        mock_get_components.side_effect = get_components_side_effect

        json_update = await get_asset_details_gr(mock_select_event)

    mock_get_entity.assert_called_once_with(mock_session, 1)
    assert mock_get_components.call_count == 2 # Called for each registered type

    expected_json = {
        "FilePropertiesComponent": [{"original_filename": "test.jpg", "mime_type": "image/jpeg"}],
        "ImageDimensionsComponent": [{"width": 100, "height": 200}]
    }
    assert json_update.value == expected_json
    assert f"Components for Asset ID: 1" in json_update.label


async def test_get_asset_details_gr_db_error(mock_active_world):
    """Test get_asset_details_gr when a database error occurs during component fetching."""
    mock_select_event = MagicMock(spec=gr.SelectData)
    mock_select_event.value = 1
    mock_select_event.index = (0,0)

    mock_session = mock_active_world.get_db_session.return_value.__aenter__.return_value
    mock_entity = MagicMock(spec=Entity); mock_entity.id = 1

    with patch("dam.gradio_ui.ecs_service.get_entity", AsyncMock(return_value=mock_entity)), \
         patch("dam.gradio_ui.ecs_service.get_components", AsyncMock(side_effect=Exception("DB Component Error"))), \
         patch("dam.gradio_ui.REGISTERED_COMPONENT_TYPES", [FilePropertiesComponent]): # Use one type for simplicity

        json_update = await get_asset_details_gr(mock_select_event)

    assert "Error: Could not fetch details" in json_update.value["error"]
    assert "DB Component Error" in json_update.value["error"]


@pytest.fixture
def mock_file_obj():
    """Fixture to create a mock Gradio File object."""
    file_mock = MagicMock(spec=gr.File) # Use spec for gr.File if it's a class, else just MagicMock
    # Gradio's File component when type="filepath" provides `name` attribute as temp path
    file_mock.name = "/tmp/fake_uploaded_file.jpg"
    return file_mock

async def test_add_assets_ui_no_active_world(mock_file_obj):
    """Test add_assets_ui when no world is active."""
    with patch("dam.gradio_ui._active_world", None):
        status = await add_assets_ui(files=[mock_file_obj], no_copy=False)
    assert status == "Error: No active world selected. Please select a world first."

async def test_add_assets_ui_no_files():
    """Test add_assets_ui when no files are provided."""
    status = await add_assets_ui(files=[], no_copy=False)
    assert status == "Info: No files selected to add."

async def test_add_assets_ui_success_copy(mock_active_world, mock_file_obj):
    """Test add_assets_ui successfully dispatches copy ingestion."""
    # Mock file_operations.get_file_properties
    with patch("dam.gradio_ui.file_operations.get_file_properties") as mock_get_props, \
         patch("pathlib.Path.exists") as mock_path_exists, \
         patch("pathlib.Path.is_file") as mock_path_is_file:

        mock_get_props.return_value = ("test.jpg", 1024, "image/jpeg")
        mock_path_exists.return_value = True # Simulate file exists at temp path
        mock_path_is_file.return_value = True  # Simulate it's a file

        status = await add_assets_ui(files=[mock_file_obj], no_copy=False)

    mock_get_props.assert_called_once_with(Path("/tmp/fake_uploaded_file.jpg"))
    mock_active_world.dispatch_event.assert_called_once()
    dispatched_event = mock_active_world.dispatch_event.call_args[0][0]
    assert isinstance(dispatched_event, AssetFileIngestionRequested)
    assert dispatched_event.original_filename == "test.jpg"
    mock_active_world.execute_stage.assert_called_once_with(ecs_service.SystemStage.METADATA_EXTRACTION)
    assert "Success: Dispatched ingestion for 'test.jpg' (Type: Copy)." in status
    assert "Operation Summary: 1 succeeded, 0 failed." in status


async def test_add_assets_ui_success_no_copy(mock_active_world, mock_file_obj):
    """Test add_assets_ui successfully dispatches reference ingestion."""
    with patch("dam.gradio_ui.file_operations.get_file_properties") as mock_get_props, \
         patch("pathlib.Path.exists") as mock_path_exists, \
         patch("pathlib.Path.is_file") as mock_path_is_file:

        mock_get_props.return_value = ("ref.txt", 512, "text/plain")
        mock_path_exists.return_value = True
        mock_path_is_file.return_value = True

        status = await add_assets_ui(files=[mock_file_obj], no_copy=True)

    mock_get_props.assert_called_once_with(Path("/tmp/fake_uploaded_file.jpg")) # Path is still the temp path
    mock_active_world.dispatch_event.assert_called_once()
    dispatched_event = mock_active_world.dispatch_event.call_args[0][0]
    assert isinstance(dispatched_event, AssetReferenceIngestionRequested)
    assert dispatched_event.original_filename == "ref.txt"
    assert "Success: Dispatched ingestion for 'ref.txt' (Type: Reference (no copy))." in status
    assert "Operation Summary: 1 succeeded, 0 failed." in status


async def test_add_assets_ui_file_processing_error(mock_active_world, mock_file_obj):
    """Test add_assets_ui when get_file_properties raises an error."""
    with patch("dam.gradio_ui.file_operations.get_file_properties", side_effect=Exception("Read error")) as mock_get_props, \
         patch("pathlib.Path.exists") as mock_path_exists, \
         patch("pathlib.Path.is_file") as mock_path_is_file:

        mock_path_exists.return_value = True
        mock_path_is_file.return_value = True

        status = await add_assets_ui(files=[mock_file_obj], no_copy=False)

    mock_get_props.assert_called_once()
    mock_active_world.dispatch_event.assert_not_called() # Should not be called if props fail
    assert "Error processing file 'fake_uploaded_file.jpg': Read error" in status
    assert "Operation Summary: 0 succeeded, 1 failed." in status

async def test_add_assets_ui_temp_file_not_found(mock_active_world, mock_file_obj):
    """Test add_assets_ui when the temporary uploaded file does not exist."""
    with patch("pathlib.Path.exists") as mock_path_exists:
        mock_path_exists.return_value = False # Simulate file does not exist at temp path

        status = await add_assets_ui(files=[mock_file_obj], no_copy=False)

    assert f"Error: File 'fake_uploaded_file.jpg' (temp path: {mock_file_obj.name}) not found" in status
    assert "Operation Summary: 0 succeeded, 1 failed." in status
    mock_active_world.dispatch_event.assert_not_called()


async def test_find_by_hash_ui_no_active_world():
    """Test find_by_hash_ui when no world is active."""
    with patch("dam.gradio_ui._active_world", None):
        json_update = await find_by_hash_ui(hash_value="somehash", hash_type="sha256")
    assert json_update.value == {"error": "Error: No active world selected."}


async def test_find_by_hash_ui_empty_hash_value():
    """Test find_by_hash_ui with an empty hash value."""
    json_update = await find_by_hash_ui(hash_value="  ", hash_type="sha256")
    assert json_update.value == {"error": "Error: Hash value cannot be empty."}


async def test_find_by_hash_ui_success(mock_active_world):
    """Test find_by_hash_ui successfully finds an asset."""
    mock_event_future = asyncio.Future()
    mock_event_future.set_result({"entity_id": 1, "original_filename": "found.jpg"})

    # Patch the loop used by the function to control the future
    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_loop.create_future.return_value = mock_event_future
        mock_get_loop.return_value = mock_loop

        json_update = await find_by_hash_ui(hash_value="testhash", hash_type="sha256")

    mock_active_world.dispatch_event.assert_called_once()
    dispatched_event = mock_active_world.dispatch_event.call_args[0][0]
    assert isinstance(dispatched_event, FindEntityByHashQuery)
    assert dispatched_event.hash_value == "testhash"
    assert dispatched_event.hash_type == "sha256"
    assert dispatched_event.result_future == mock_event_future

    assert json_update.value == {"entity_id": 1, "original_filename": "found.jpg"}
    assert "Asset Found" in json_update.label


async def test_find_by_hash_ui_not_found(mock_active_world):
    """Test find_by_hash_ui when no asset is found."""
    mock_event_future = asyncio.Future()
    mock_event_future.set_result(None) # Simulate asset not found

    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_loop.create_future.return_value = mock_event_future
        mock_get_loop.return_value = mock_loop
        json_update = await find_by_hash_ui(hash_value="unknownhash", hash_type="md5")

    assert "Info: No asset found for hash unknownhash (Type: md5)." in json_update.value["info"]


async def test_find_by_hash_ui_timeout(mock_active_world):
    """Test find_by_hash_ui when the query times out."""
    mock_event_future = asyncio.Future() # Future that will never be resolved for timeout

    with patch("asyncio.get_running_loop") as mock_get_loop, \
         patch("asyncio.wait_for", side_effect=asyncio.TimeoutError): # Mock wait_for to raise TimeoutError
        mock_loop = MagicMock()
        mock_loop.create_future.return_value = mock_event_future
        mock_get_loop.return_value = mock_loop

        json_update = await find_by_hash_ui(hash_value="longqueryhash", hash_type="sha256")

    assert "Error: Query timed out for hash longqueryhash." in json_update.value["error"]


async def test_find_by_hash_ui_dispatch_error(mock_active_world):
    """Test find_by_hash_ui when dispatch_event itself raises an error."""
    mock_active_world.dispatch_event.side_effect = Exception("Dispatch failed")

    json_update = await find_by_hash_ui(hash_value="somehash", hash_type="sha256")

    assert "Error: Could not find asset by hash. Dispatch failed" in json_update.value["error"]


async def test_export_world_ui_no_active_world():
    """Test export_world_ui when no world is active."""
    with patch("dam.gradio_ui._active_world", None):
        status = await export_world_ui(export_path="/path/to/export.json")
    assert status == "Error: No active world selected for export."


async def test_export_world_ui_no_export_path():
    """Test export_world_ui with no export path provided."""
    status = await export_world_ui(export_path="  ") # Empty or whitespace only
    assert status == "Error: Export file path must be provided."


async def test_export_world_ui_parent_dir_not_exist():
    """Test export_world_ui when parent directory of export path does not exist."""
    with patch("pathlib.Path.is_absolute", return_value=False), \
         patch("pathlib.Path.parent") as mock_parent:
        mock_parent.exists.return_value = False
        mock_parent.__ne__.return_value = True # Ensure parent is not Path(".")

        status = await export_world_ui(export_path="non_existent_parent/export.json")
    assert "Error: Parent directory 'non_existent_parent' for export does not exist." in status


async def test_export_world_ui_path_is_directory():
    """Test export_world_ui when export path is a directory."""
    with patch("pathlib.Path.is_dir", return_value=True) as mock_is_dir:
        status = await export_world_ui(export_path="/path/to/existing_dir")
    mock_is_dir.assert_called_once() # is_dir is checked by the function
    assert "Error: Export path '/path/to/existing_dir' is a directory, must be a file path." in status


async def test_export_world_ui_success(mock_active_world):
    """Test export_world_ui successfully calls the service."""
    with patch("dam.gradio_ui.asyncio.to_thread") as mock_to_thread, \
         patch("pathlib.Path.is_absolute", return_value=True), \
         patch("pathlib.Path.is_dir", return_value=False): # Ensure it's not a dir

        # mock_to_thread needs to be an async function for await
        async def async_wrapper(*args, **kwargs):
            # Simulate the actual function call if needed or just return None
            # For this test, we just want to ensure it's called.
            return None
        mock_to_thread.side_effect = async_wrapper # Make it awaitable

        status = await export_world_ui(export_path="/path/to/export_file.json")

    mock_to_thread.assert_called_once()
    # Check args passed to to_thread, which includes the service function and its args
    call_args = mock_to_thread.call_args[0]
    assert call_args[0] == world_service.export_ecs_world_to_json
    assert call_args[1] == mock_active_world # First arg to service func
    assert call_args[2] == Path("/path/to/export_file.json") # Second arg

    assert "Success: World 'test_world' exported to '/path/to/export_file.json'." in status


async def test_export_world_ui_service_error(mock_active_world):
    """Test export_world_ui when the world_service call raises an error."""
    with patch("dam.gradio_ui.asyncio.to_thread") as mock_to_thread, \
         patch("pathlib.Path.is_absolute", return_value=True), \
         patch("pathlib.Path.is_dir", return_value=False):

        async def async_wrapper_with_error(*args, **kwargs):
            raise Exception("Service export failed")
        mock_to_thread.side_effect = async_wrapper_with_error # Make it awaitable and raise error

        status = await export_world_ui(export_path="/path/to/another_export.json")

    assert "Error: Could not export world 'test_world'. Service export failed" in status
