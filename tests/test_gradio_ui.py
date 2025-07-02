import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

import gradio as gr
from pathlib import Path

from dam.gradio_ui import (
    set_active_world_and_refresh_dropdowns,
    list_assets_gr,
    get_asset_details_gr,
    add_assets_ui,
    find_by_hash_ui,
    export_world_ui,
)

from dam.core.world import World
from dam.services import ecs_service, world_service, file_operations
from dam.core.events import FindEntityByHashQuery, AssetFileIngestionRequested, AssetReferenceIngestionRequested
from dam.models.core.entity import Entity
from dam.models.properties import FilePropertiesComponent


pytestmark = pytest.mark.asyncio

@pytest.fixture
def active_test_world(test_world_with_db_session, monkeypatch):
    current_test_world = test_world_with_db_session
    monkeypatch.setattr("dam.gradio_ui._active_world", current_test_world)

    def _get_current_world_mock(world_name: Optional[str]):
        if world_name == current_test_world.name:
            return current_test_world
        from dam.gradio_ui import get_current_world as original_get_current_world
        return original_get_current_world(world_name)

    monkeypatch.setattr("dam.gradio_ui.get_current_world", _get_current_world_mock)
    return current_test_world

# --- Tests for set_active_world_and_refresh_dropdowns ---
async def test_set_active_world_valid_world(active_test_world, monkeypatch):
    mock_session_execute_result = MagicMock()
    mock_scalars_obj = MagicMock()
    mock_scalars_obj.all.return_value = ["image/jpeg", "application/pdf"]
    mock_session_execute_result.scalars.return_value = mock_scalars_obj

    mock_session_itself = AsyncMock()
    mock_session_itself.execute = AsyncMock(return_value=mock_session_execute_result)

    async def mock_async_context_manager(*args, **kwargs):
        return mock_session_itself

    monkeypatch.setattr(active_test_world, "get_db_session", MagicMock(return_value=MagicMock(__aenter__=mock_async_context_manager, __aexit__=AsyncMock(return_value=False))))

    with patch("dam.gradio_ui.get_transcode_profile_choices", new_callable=AsyncMock) as mock_get_tp, \
         patch("dam.gradio_ui.get_evaluation_run_choices", new_callable=AsyncMock) as mock_get_er:
        mock_get_tp.return_value = ["default_profile"]
        mock_get_er.return_value = ["default_run"]

        status, mime_dd, tp_dd, er_exec_dd, er_report_dd = await set_active_world_and_refresh_dropdowns(active_test_world.name)

    assert f"Success: World '{active_test_world.name}' selected." in status
    assert "MIME types refreshed." in status

    assert isinstance(mime_dd, gr.Dropdown)
    dropdown_values = [val for label, val in mime_dd.choices]
    assert "" in dropdown_values
    assert "image/jpeg" in dropdown_values
    assert "application/pdf" in dropdown_values
    assert mime_dd.value == ""

    assert isinstance(tp_dd, gr.Dropdown)
    assert tp_dd.choices == [("default_profile", "default_profile")]
    assert isinstance(er_exec_dd, gr.Dropdown)
    assert er_exec_dd.choices == [("default_run", "default_run")]
    assert isinstance(er_report_dd, gr.Dropdown)
    assert er_report_dd.choices == [("default_run", "default_run")]

async def test_set_active_world_invalid_world_name(monkeypatch):
    def _get_current_world_mock_invalid(world_name: Optional[str]):
        return None
    monkeypatch.setattr("dam.gradio_ui.get_current_world", _get_current_world_mock_invalid)
    monkeypatch.setattr("dam.gradio_ui._active_world", None)
    status, mime_dd, tp_dd, er_exec_dd, er_report_dd = await set_active_world_and_refresh_dropdowns("invalid_world_name")
    assert status == "Error: Failed to select world or no valid world chosen."
    assert mime_dd.choices == [("", "")]
    assert tp_dd.choices == [("Info: Select world first or refresh", "Info: Select world first or refresh")]
    assert er_exec_dd.choices == [("Info: Select world first or refresh", "Info: Select world first or refresh")]
    assert er_report_dd.choices == [("Info: Select world first or refresh", "Info: Select world first or refresh")]

async def test_set_active_world_mime_type_load_failure(active_test_world, monkeypatch):
    mock_session_cm = AsyncMock()
    mock_session_obj = AsyncMock()
    mock_session_obj.execute = AsyncMock(side_effect=Exception("DB error during MIME fetch"))
    async def cm_aenter(*args, **kwargs): return mock_session_obj
    mock_session_cm.__aenter__ = cm_aenter
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(active_test_world, "get_db_session", MagicMock(return_value=mock_session_cm))

    with patch("dam.gradio_ui.get_transcode_profile_choices", new_callable=AsyncMock) as mock_get_tp, \
         patch("dam.gradio_ui.get_evaluation_run_choices", new_callable=AsyncMock) as mock_get_er:
        mock_get_tp.return_value = ["default_profile"]
        mock_get_er.return_value = ["default_run"]
        status, mime_dd, _, _, _ = await set_active_world_and_refresh_dropdowns(active_test_world.name)

    assert f"Success: World '{active_test_world.name}' selected." in status
    assert "Error loading MIME types: DB error during MIME fetch" in status
    assert mime_dd.choices == [("", ""), ("Error: Could not load MIME types", "Error: Could not load MIME types")]


# --- Tests for list_assets_gr ---
async def test_list_assets_gr_no_active_world(monkeypatch):
    monkeypatch.setattr("dam.gradio_ui._active_world", None)
    df_update, status = await list_assets_gr(filename_filter="", mime_type_filter="")
    assert status == "Error: No active world selected. Please select a world first."
    assert df_update.value['data'] == []

async def test_list_assets_gr_success_no_filters(active_test_world):
    async with active_test_world.get_db_session() as session:
        e1 = await ecs_service.create_entity(session)
        await ecs_service.add_component_to_entity(session, e1.id, FilePropertiesComponent(original_filename="file1.jpg", mime_type="image/jpeg"))
        e2 = await ecs_service.create_entity(session)
        await ecs_service.add_component_to_entity(session, e2.id, FilePropertiesComponent(original_filename="file2.png", mime_type="image/png"))
        await session.commit()
    df_update, status = await list_assets_gr(filename_filter="", mime_type_filter="")
    assert "Info: Displaying 2 of 2 assets (Page 1)." in status
    actual_data_as_tuples = [tuple(row) for row in df_update.value["data"]]
    expected_data_as_tuples = sorted([(e1.id, "file1.jpg", "image/jpeg"), (e2.id, "file2.png", "image/png")], key=lambda x: x[0])
    assert sorted(actual_data_as_tuples, key=lambda x: x[0]) == expected_data_as_tuples

async def test_list_assets_gr_with_filters(active_test_world):
    async with active_test_world.get_db_session() as session:
        e1 = await ecs_service.create_entity(session)
        await ecs_service.add_component_to_entity(session, e1.id, FilePropertiesComponent(original_filename="test_file.jpg", mime_type="image/jpeg"))
        e2 = await ecs_service.create_entity(session)
        await ecs_service.add_component_to_entity(session, e2.id, FilePropertiesComponent(original_filename="another.txt", mime_type="text/plain"))
        await session.commit()
    df_update, status = await list_assets_gr(filename_filter="test_file", mime_type_filter="image/jpeg")
    assert "Info: Displaying 1 of 1 assets (Page 1)." in status
    actual_data_as_tuples = [tuple(row) for row in df_update.value["data"]]
    expected_data = [(e1.id, "test_file.jpg", "image/jpeg")]
    assert actual_data_as_tuples == expected_data

async def test_list_assets_gr_pagination(active_test_world):
    async with active_test_world.get_db_session() as session:
        for i in range(1, 26):
            e = await ecs_service.create_entity(session)
            await ecs_service.add_component_to_entity(session, e.id, FilePropertiesComponent(original_filename=f"file{i:02d}.txt", mime_type="text/plain"))
        await session.commit()
    df_update, status = await list_assets_gr(filename_filter="", mime_type_filter="", current_page=2)
    assert "Info: Displaying 5 of 25 assets (Page 2)." in status
    assert len(df_update.value["data"]) == 5
    assert df_update.value["data"][0][1] == "file21.txt"

async def test_list_assets_gr_no_results(active_test_world):
    df_update, status = await list_assets_gr(filename_filter="nonexistent", mime_type_filter="")
    assert "Info: No assets found matching criteria." in status
    assert len(df_update.value["data"]) == 0

async def test_list_assets_gr_db_error(active_test_world, monkeypatch):
    mock_session_cm = AsyncMock(); mock_session_obj = AsyncMock()
    mock_session_obj.execute = AsyncMock(side_effect=Exception("Simulated DB connection failed"))
    async def cm_aenter(*args, **kwargs): return mock_session_obj
    mock_session_cm.__aenter__ = cm_aenter; mock_session_cm.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(active_test_world, "get_db_session", MagicMock(return_value=mock_session_cm))
    df_update, status = await list_assets_gr(filename_filter="", mime_type_filter="")
    assert "Error: Could not load assets. Details: Simulated DB connection failed" in status
    assert df_update.value['data'] == []

# --- Tests for get_asset_details_gr ---
async def test_get_asset_details_gr_no_active_world(monkeypatch):
    monkeypatch.setattr("dam.gradio_ui._active_world", None)
    mock_select_event = MagicMock(spec=gr.SelectData); mock_select_event.value = 1; mock_select_event.index = (0,0)
    json_update = await get_asset_details_gr(mock_select_event)
    assert json_update.value == {"error": "Error: No active world selected."}

async def test_get_asset_details_gr_invalid_selection_event(active_test_world):
    mock_select_event = MagicMock(spec=gr.SelectData); mock_select_event.value = None; mock_select_event.index = (None, None)
    json_update = await get_asset_details_gr(mock_select_event)
    assert json_update.value == {"info": "Info: Click on an Asset ID in the table above to view its details."}

async def test_get_asset_details_gr_asset_not_found(active_test_world):
    mock_select_event = MagicMock(spec=gr.SelectData); mock_select_event.value = 999; mock_select_event.index = (0,0)
    json_update = await get_asset_details_gr(mock_select_event)
    assert json_update.value == {"error": "Error: Asset ID 999 not found."}

async def test_get_asset_details_gr_success(active_test_world):
    mock_select_event = MagicMock(spec=gr.SelectData)

    async with active_test_world.get_db_session() as session:
        entity = await ecs_service.create_entity(session)
        mock_select_event.value = entity.id
        mock_select_event.index = (0,0)
        await ecs_service.add_component_to_entity(session, entity.id, FilePropertiesComponent(original_filename="f.jpg", mime_type="image/jpeg"))
        await session.commit()

    with patch("dam.gradio_ui.REGISTERED_COMPONENT_TYPES", [FilePropertiesComponent]):
        json_update = await get_asset_details_gr(mock_select_event)

    assert "FilePropertiesComponent" in json_update.value
    assert len(json_update.value["FilePropertiesComponent"]) == 1
    fpc_data = json_update.value["FilePropertiesComponent"][0]
    assert fpc_data["original_filename"] == "f.jpg"
    assert fpc_data["mime_type"] == "image/jpeg"
    assert fpc_data["entity_id"] == entity.id

async def test_get_asset_details_gr_db_error(active_test_world, monkeypatch):
    mock_select_event = MagicMock(spec=gr.SelectData); mock_select_event.value = 1; mock_select_event.index = (0,0)
    async with active_test_world.get_db_session() as session:
        await ecs_service.create_entity(session, entity_id=1); await session.commit()

    async_mock_get_components = AsyncMock(side_effect=Exception("DB Comp Error"))
    monkeypatch.setattr(ecs_service, "get_components", async_mock_get_components)

    with patch("dam.gradio_ui.REGISTERED_COMPONENT_TYPES", [FilePropertiesComponent]):
        json_update = await get_asset_details_gr(mock_select_event)

    assert "Error: Could not fetch details" in json_update.value["error"]
    assert "DB Comp Error" in json_update.value["error"]


# --- Tests for add_assets_ui ---
@pytest.fixture
def mock_file_obj():
    file_mock = MagicMock(); file_mock.name = "/tmp/fake_uploaded_file.jpg"
    return file_mock

async def test_add_assets_ui_no_active_world(mock_file_obj, monkeypatch):
    monkeypatch.setattr("dam.gradio_ui._active_world", None)
    status = await add_assets_ui(files=[mock_file_obj], no_copy=False)
    assert status == "Error: No active world selected. Please select a world first."

async def test_add_assets_ui_no_files(active_test_world):
    status = await add_assets_ui(files=[], no_copy=False)
    assert status == "Info: No files selected to add."

async def test_add_assets_ui_success_copy(active_test_world, mock_file_obj, monkeypatch):
    with patch("dam.gradio_ui.file_operations.get_file_properties", return_value=("t.jpg",1, "i/j")) as m_gp, \
         patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):

        mock_dispatch = AsyncMock()
        mock_execute_stage = AsyncMock()
        monkeypatch.setattr(active_test_world, "dispatch_event", mock_dispatch)
        monkeypatch.setattr(active_test_world, "execute_stage", mock_execute_stage)

        status = await add_assets_ui(files=[mock_file_obj], no_copy=False)

    m_gp.assert_called_once_with(Path("/tmp/fake_uploaded_file.jpg"))
    mock_dispatch.assert_called_once()
    assert isinstance(mock_dispatch.call_args[0][0], AssetFileIngestionRequested)
    mock_execute_stage.assert_called_once_with(ecs_service.SystemStage.METADATA_EXTRACTION)
    assert "Success: Dispatched ingestion for 't.jpg' (Type: Copy)." in status

async def test_add_assets_ui_success_no_copy(active_test_world, mock_file_obj, monkeypatch):
    with patch("dam.gradio_ui.file_operations.get_file_properties", return_value=("r.txt",1,"t/p")) as m_gp, \
         patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        mock_dispatch = AsyncMock()
        mock_execute_stage = AsyncMock()
        monkeypatch.setattr(active_test_world, "dispatch_event", mock_dispatch)
        monkeypatch.setattr(active_test_world, "execute_stage", mock_execute_stage)
        status = await add_assets_ui(files=[mock_file_obj], no_copy=True)
    m_gp.assert_called_once()
    assert isinstance(mock_dispatch.call_args[0][0], AssetReferenceIngestionRequested)
    assert "Success: Dispatched ingestion for 'r.txt' (Type: Reference (no copy))." in status

async def test_add_assets_ui_file_processing_error(active_test_world, mock_file_obj, monkeypatch):
    with patch("dam.gradio_ui.file_operations.get_file_properties", side_effect=Exception("Read error")) as m_gp, \
         patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        mock_dispatch = AsyncMock()
        monkeypatch.setattr(active_test_world, "dispatch_event", mock_dispatch)
        status = await add_assets_ui(files=[mock_file_obj], no_copy=False)
    m_gp.assert_called_once()
    mock_dispatch.assert_not_called()
    assert "Error processing file 'fake_uploaded_file.jpg': Read error" in status

async def test_add_assets_ui_temp_file_not_found(active_test_world, mock_file_obj, monkeypatch):
    with patch("pathlib.Path.exists", return_value=False):
        mock_dispatch = AsyncMock()
        monkeypatch.setattr(active_test_world, "dispatch_event", mock_dispatch)
        status = await add_assets_ui(files=[mock_file_obj], no_copy=False)
    assert f"Error: File 'fake_uploaded_file.jpg' (temp path: {mock_file_obj.name}) not found" in status
    mock_dispatch.assert_not_called()

# --- Tests for find_by_hash_ui ---
async def test_find_by_hash_ui_no_active_world(monkeypatch):
    monkeypatch.setattr("dam.gradio_ui._active_world", None)
    json_update = await find_by_hash_ui(hash_value="h", hash_type="sha256")
    assert json_update.value == {"error": "Error: No active world selected."}

async def test_find_by_hash_ui_empty_hash_value(active_test_world):
    json_update = await find_by_hash_ui(hash_value=" ", hash_type="sha256")
    assert json_update.value == {"error": "Error: Hash value cannot be empty."}

async def test_find_by_hash_ui_success(active_test_world):
    hash_val = "testhashsuccessful"
    expected_entity_id = -1
    async with active_test_world.get_db_session() as session:
        entity = await ecs_service.create_entity(session)
        expected_entity_id = entity.id
        await ecs_service.add_component_to_entity(session, entity.id, FilePropertiesComponent(original_filename="hashed_file.jpg", entity_id=entity.id))
        from dam.models.hashes import ContentHashSHA256Component
        await ecs_service.add_component_to_entity(session, entity.id, ContentHashSHA256Component(hash_value=hash_val, entity_id=entity.id))
        await session.commit()

    json_update = await find_by_hash_ui(hash_value=hash_val, hash_type="sha256")

    assert "Asset Found" in json_update.label
    assert json_update.value["entity_id"] == expected_entity_id
    assert "FilePropertiesComponent" in json_update.value["components"]
    assert json_update.value["components"]["FilePropertiesComponent"][0]["original_filename"] == "hashed_file.jpg"
    assert "ContentHashSHA256Component" in json_update.value["components"]
    assert json_update.value["components"]["ContentHashSHA256Component"][0]["hash_value"] == hash_val


async def test_find_by_hash_ui_not_found(active_test_world):
    json_update = await find_by_hash_ui(hash_value="unknownhash", hash_type="md5")
    assert "Info: No asset found" in json_update.value["info"]

async def test_find_by_hash_ui_timeout(active_test_world, monkeypatch):
    hanging_future = asyncio.Future()
    async def mock_dispatch_with_hanging_future(event): event.result_future = hanging_future
    monkeypatch.setattr(active_test_world, "dispatch_event", AsyncMock(side_effect=mock_dispatch_with_hanging_future))
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        json_update = await find_by_hash_ui(hash_value="longqueryhash", hash_type="sha256")
    assert "Error: Query timed out" in json_update.value["error"]

async def test_find_by_hash_ui_dispatch_error(active_test_world, monkeypatch):
    monkeypatch.setattr(active_test_world, "dispatch_event", AsyncMock(side_effect=Exception("Dispatch failed")))
    json_update = await find_by_hash_ui(hash_value="somehash", hash_type="sha256")
    assert "Error: Could not find asset by hash. Dispatch failed" in json_update.value["error"]

# --- Tests for export_world_ui ---
async def test_export_world_ui_no_active_world(monkeypatch):
    monkeypatch.setattr("dam.gradio_ui._active_world", None)
    status = await export_world_ui(export_path="p.json")
    assert status == "Error: No active world selected for export."

async def test_export_world_ui_no_export_path(active_test_world):
    status = await export_world_ui(export_path=" ")
    assert status == "Error: Export file path must be provided."

async def test_export_world_ui_parent_dir_not_exist(active_test_world):
    with patch("pathlib.Path.is_absolute", return_value=False), \
         patch("pathlib.Path.parent") as mp:
        mp.exists.return_value = False; mp.__ne__.return_value = True # parent is not "."
        status = await export_world_ui(export_path="parent/p.json")
    assert "Error: Parent directory 'parent' for export does not exist." in status

async def test_export_world_ui_path_is_directory(active_test_world):
    with patch("pathlib.Path.is_dir", return_value=True) as mid:
        status = await export_world_ui(export_path="/dir")
    mid.assert_called_once()
    assert "Error: Export path '/dir' is a directory" in status

async def test_export_world_ui_success(active_test_world):
    with patch("dam.gradio_ui.world_service.export_ecs_world_to_json") as mock_export_service, \
         patch("pathlib.Path.is_absolute", return_value=True), \
         patch("pathlib.Path.is_dir", return_value=False):
        status = await export_world_ui(export_path="/p.json")
    mock_export_service.assert_called_once_with(active_test_world, Path("/p.json"))
    assert f"Success: World '{active_test_world.name}' exported" in status

async def test_export_world_ui_service_error(active_test_world):
    with patch("dam.gradio_ui.world_service.export_ecs_world_to_json", side_effect=Exception("Service fail")), \
         patch("pathlib.Path.is_absolute", return_value=True), \
         patch("pathlib.Path.is_dir", return_value=False):
        status = await export_world_ui(export_path="/p.json")
    assert f"Error: Could not export world '{active_test_world.name}'. Service fail" in status
