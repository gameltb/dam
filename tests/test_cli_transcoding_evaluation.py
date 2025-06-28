import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock, ANY # Added ANY
import json # For manipulating JSON strings for config
import os # For os.environ interaction, though pytest.os.environ is preferred in fixtures

from dam.cli import app
from dam.core.world import World
from dam.core.config import Settings
from dam.models.core.entity import Entity
from dam.services import transcode_service # For type hints
# Aliasing ecs_service to avoid potential conflicts if this test module grows complex
from dam.services import ecs_service as dam_ecs_service_module
from dam.systems import evaluation_systems
from dam.models.conceptual.evaluation_result_component import EvaluationResultComponent


runner = CliRunner()

cli_test_settings_dict = {
    "DAM_WORLDS_CONFIG": '''{
        "cli_test_world": {
            "DATABASE_URL": "sqlite+aiosqlite:///./test_cli_dam.db",
            "ASSET_STORAGE_PATH": "./test_cli_dam_storage"
        }
    }''',
    "DAM_DEFAULT_WORLD_NAME": "cli_test_world",
    "TESTING_MODE": "True",
    "DAM_TRANSCODING_TEMP_DIR": "temp/test_cli_transcodes"
}

@pytest.fixture(scope="module", autouse=True)
def setup_cli_test_environment(tmp_path_factory):
    storage_base = tmp_path_factory.mktemp("cli_dam_storage")
    transcode_temp = tmp_path_factory.mktemp("cli_transcodes_temp")

    world_conf_json_str = cli_test_settings_dict["DAM_WORLDS_CONFIG"]
    world_conf_data = json.loads(world_conf_json_str)
    world_conf_data["cli_test_world"]["ASSET_STORAGE_PATH"] = str(storage_base)

    db_file = Path("./test_cli_dam.db")
    db_file.unlink(missing_ok=True)

    original_env = {}
    env_to_set = {
        "DAM_WORLDS_CONFIG": json.dumps(world_conf_data),
        "DAM_DEFAULT_WORLD_NAME": cli_test_settings_dict["DAM_DEFAULT_WORLD_NAME"],
        "TESTING_MODE": cli_test_settings_dict["TESTING_MODE"],
        "DAM_TRANSCODING_TEMP_DIR": str(transcode_temp)
    }

    for key, value in env_to_set.items():
        if key in os.environ:
            original_env[key] = os.environ[key]
        os.environ[key] = value

    from dam.core import config as dam_config_module
    dam_config_module.settings = dam_config_module.Settings()


    result_setup = runner.invoke(app, ["--world", "cli_test_world", "setup-db"])
    if result_setup.exit_code != 0:
        print("CLI setup-db output on failure:", result_setup.stdout)
        print("Error output:", result_setup.stderr)
    assert result_setup.exit_code == 0, f"CLI setup-db failed: {result_setup.stdout}"

    yield

    for key, value in env_to_set.items():
        if key in original_env:
            os.environ[key] = original_env[key]
        else:
            del os.environ[key]

    dam_config_module.settings = dam_config_module.Settings()

    db_file.unlink(missing_ok=True)


@patch("dam.services.transcode_service.create_transcode_profile", new_callable=AsyncMock)
def test_cli_transcode_profile_create(mock_create_profile: AsyncMock):
    mock_profile_entity = Entity(id=123)
    mock_create_profile.return_value = mock_profile_entity

    result = runner.invoke(app, [
        "--world", "cli_test_world",
        "transcode", "profile-create",
        "--name", "cli_test_profile", "--tool", "ffmpeg",
        "--params", "-crf 22 {output}", "--format", "mkv",
        "--desc", "A test profile from CLI"
    ])
    assert result.exit_code == 0
    assert "Transcode profile 'cli_test_profile' (Entity ID: 123) created successfully" in result.stdout
    mock_create_profile.assert_called_once()
    call_args = mock_create_profile.call_args[1]
    assert call_args['profile_name'] == "cli_test_profile"
    assert isinstance(call_args['world'], World)
    assert call_args['world'].name == "cli_test_world"


@patch("dam.services.transcode_service.apply_transcode_profile", new_callable=AsyncMock)
@patch("dam.cli.dam_ecs_service.find_entity_id_by_hash", new_callable=AsyncMock)
@patch("dam.cli.dam_ecs_service.get_component_for_target_entity", new_callable=AsyncMock)
def test_cli_transcode_apply(mock_get_fpc: AsyncMock, mock_find_by_hash: AsyncMock, mock_apply: AsyncMock):
    mock_transcoded_entity = Entity(id=456)
    mock_fpc_data = {"original_filename": "transcoded_via_cli.mkv", "file_size_bytes": 5000}

    mock_apply.return_value = mock_transcoded_entity

    mock_fpc_instance = MagicMock()
    mock_fpc_instance.original_filename = mock_fpc_data["original_filename"]
    mock_fpc_instance.file_size_bytes = mock_fpc_data["file_size_bytes"]
    mock_get_fpc.return_value = mock_fpc_instance

    result = runner.invoke(app, [
        "--world", "cli_test_world", "transcode", "apply",
        "--asset", "1", "--profile", "10"
    ])
    assert result.exit_code == 0
    assert "Transcoding successful. New transcoded asset Entity ID: 456" in result.stdout
    assert f"New Filename: {mock_fpc_data['original_filename']}" in result.stdout
    mock_apply.assert_called_once()
    call_args_apply = mock_apply.call_args[1]
    assert call_args_apply['source_asset_entity_id'] == 1
    assert call_args_apply['profile_entity_id'] == 10
    mock_get_fpc.assert_called_once()

    mock_apply.reset_mock(); mock_get_fpc.reset_mock(); mock_find_by_hash.reset_mock()
    mock_find_by_hash.return_value = 2

    result_hash = runner.invoke(app, [
        "--world", "cli_test_world", "transcode", "apply",
        "--asset", "somehashvalue", "--profile", "someprofilename"
    ])
    assert result_hash.exit_code == 0
    mock_find_by_hash.assert_called_once_with(ANY, hash_value="somehashvalue", hash_type="sha256")
    mock_apply.assert_called_once()
    call_args_apply_hash = mock_apply.call_args[1]
    assert call_args_apply_hash['source_asset_entity_id'] == 2
    assert call_args_apply_hash['profile_entity_id'] == "someprofilename"
    mock_get_fpc.assert_called_once()


@patch("dam.systems.evaluation_systems.create_evaluation_run_concept", new_callable=AsyncMock)
def test_cli_eval_run_create(mock_create_run: AsyncMock):
    mock_run_entity = Entity(id=789)
    mock_create_run.return_value = mock_run_entity

    result = runner.invoke(app, [
        "--world", "cli_test_world", "evaluate", "run-create",
        "--name", "cli_eval_run", "--desc", "A test run from CLI"
    ])
    assert result.exit_code == 0
    assert "Evaluation run 'cli_eval_run' (Entity ID: 789) created successfully" in result.stdout
    mock_create_run.assert_called_once()


@patch("dam.systems.evaluation_systems.execute_evaluation_run", new_callable=AsyncMock)
@patch("dam.cli.dam_ecs_service.find_entity_id_by_hash", new_callable=AsyncMock)
def test_cli_eval_run_execute(mock_find_hash_asset: AsyncMock, mock_execute_run: AsyncMock):
    mock_results_list = [MagicMock(spec=EvaluationResultComponent)]
    mock_execute_run.return_value = mock_results_list
    mock_find_hash_asset.return_value = 1

    result = runner.invoke(app, [
        "--world", "cli_test_world", "evaluate", "run-execute",
        "--run", "my_eval_run_name",
        "--assets", "assethash1,2",
        "--profiles", "profile_name1,15"
    ])
    assert result.exit_code == 0
    assert "Evaluation run 'my_eval_run_name' completed. Generated 1 results." in result.stdout
    mock_find_hash_asset.assert_called_once_with(ANY, hash_value="assethash1", hash_type="sha256")
    mock_execute_run.assert_called_once()
    call_args = mock_execute_run.call_args[1]
    assert call_args['source_asset_identifiers'] == ['assethash1', 2]
    assert call_args['profile_identifiers'] == ['profile_name1', 15]


@patch("dam.systems.evaluation_systems.get_evaluation_results", new_callable=AsyncMock)
def test_cli_eval_report(mock_get_results: AsyncMock):
    mock_report_data = [{
        "evaluation_run_name": "cli_reported_run",
        "original_asset_entity_id": 1, "original_asset_filename": "source.mp4",
        "profile_name": "p1", "profile_tool": "t1", "profile_params": "par1", "profile_format": "f1",
        "transcoded_asset_entity_id": 101, "transcoded_asset_filename": "transcoded1.mkv",
        "file_size_bytes": 1000, "vmaf_score": 95.5, "ssim_score": 0.99, "psnr_score": 40.1,
        "custom_metrics": {"my_metric": 10}, "notes": "Good one"
    }]
    mock_get_results.return_value = mock_report_data

    result = runner.invoke(app, [
        "--world", "cli_test_world", "evaluate", "report", "--run", "cli_reported_run_id"
    ])
    assert result.exit_code == 0
    assert "--- Evaluation Report for Run: 'cli_reported_run' ---" in result.stdout
    assert "Original Asset: source.mp4 (ID: 1)" in result.stdout
    mock_get_results.assert_called_once()


@patch("dam.services.transcode_service.apply_transcode_profile", new_callable=AsyncMock)
def test_cli_transcode_apply_tool_not_found(mock_apply_transcode: AsyncMock):
    from dam.utils.media_utils import TranscodeError as MediaUtilTranscodeError

    service_error = transcode_service.TranscodeServiceError("Transcoding failed: Tool 'xyz' not found.")
    service_error.__cause__ = MediaUtilTranscodeError("Command not found: xyz")
    mock_apply_transcode.side_effect = service_error

    result = runner.invoke(app, [
        "--world", "cli_test_world", "transcode", "apply",
        "--asset", "1", "--profile", "10"
    ])
    assert result.exit_code == 1
    assert "Error applying transcode profile: Transcoding failed: Tool 'xyz' not found." in result.stdout
    assert "Hint: Ensure the required transcoding tool" in result.stdout
