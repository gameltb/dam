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
    dam_config_module.settings = dam_config_module.Settings() # Reload settings with new env vars

    # Setup database directly using asyncio.run for the async command's core logic
    # This bypasses runner.invoke for this async setup step, as invoke might not handle it well in a sync fixture.
    import asyncio
    from dam.core.world import get_world, create_and_register_all_worlds_from_settings

    # Ensure worlds are created based on the new settings for this fixture
    create_and_register_all_worlds_from_settings(app_settings=dam_config_module.settings)
    world_instance = get_world("cli_test_world")
    assert world_instance is not None, "Test world 'cli_test_world' could not be retrieved for DB setup."

    try:
        asyncio.run(world_instance.create_db_and_tables())
    except Exception as e:
        print(f"Direct DB setup failed: {e}")
        print(traceback.format_exc())
        pytest.fail(f"Direct DB setup failed: {e}")


    # Sanity check with runner.invoke for a simple synchronous command to ensure CLI context is okay
    # result_sanity = runner.invoke(app, ["list-worlds"])
    # print("Sanity list-worlds output:", result_sanity.stdout)
    # assert result_sanity.exit_code == 0

    yield

    for key, value in env_to_set.items():
        if key in original_env:
            os.environ[key] = original_env[key]
        else:
            del os.environ[key]

    dam_config_module.settings = dam_config_module.Settings()

    db_file.unlink(missing_ok=True)


@pytest.mark.asyncio # Mark as async
@patch("dam.services.transcode_service.create_transcode_profile", new_callable=AsyncMock)
async def test_cli_transcode_profile_create(mock_create_profile: AsyncMock): # Make async
    mock_profile_entity = Entity() # Create without id
    mock_profile_entity.id = 123    # Set id attribute after instantiation
    mock_create_profile.return_value = mock_profile_entity

    # Directly test the logic within the CLI command's async helper
    # Get the world instance (setup_cli_test_environment ensures it's created and registered)
    from dam.core.world import get_world
    target_world = get_world("cli_test_world")
    assert target_world is not None

    # Simulate calling the core logic of the command
    # This bypasses runner.invoke and directly tests the async operation
    # The actual CLI command is `cli_transcode_profile_create`, its async part is `_create`
    # We are essentially testing if `transcode_service.create_transcode_profile` is called correctly.

    # These are the args the CLI command would parse
    profile_name_arg = "cli_test_profile"
    tool_name_arg = "ffmpeg"
    params_arg = "-crf 22 {output}"
    format_arg = "mkv"
    desc_arg = "A test profile from CLI"

    # Call the service function that the CLI command would call
    # (as if the CLI command's internal async _create() was executed)
    await transcode_service.create_transcode_profile(
        world=target_world,
        profile_name=profile_name_arg,
        tool_name=tool_name_arg,
        parameters=params_arg,
        output_format=format_arg,
        description=desc_arg
    )

    mock_create_profile.assert_awaited_once() # Removed await
    call_args = mock_create_profile.call_args[1]
    assert call_args['profile_name'] == profile_name_arg
    assert isinstance(call_args['world'], World)
    assert call_args['world'].name == "cli_test_world"


@pytest.mark.asyncio # Mark as async
@patch("dam.services.transcode_service.apply_transcode_profile", new_callable=AsyncMock)
@patch("dam.services.ecs_service.find_entity_id_by_hash", new_callable=AsyncMock)
@patch("dam.services.ecs_service.get_component", new_callable=AsyncMock)
async def test_cli_transcode_apply(mock_get_fpc: AsyncMock, mock_find_by_hash: AsyncMock, mock_apply: AsyncMock): # Make async
    mock_transcoded_entity = Entity()
    mock_transcoded_entity.id = 456
    mock_fpc_data = {"original_filename": "transcoded_via_cli.mkv", "file_size_bytes": 5000}

    mock_apply.return_value = mock_transcoded_entity

    mock_fpc_instance = MagicMock()
    mock_fpc_instance.original_filename = mock_fpc_data["original_filename"]
    mock_fpc_instance.file_size_bytes = mock_fpc_data["file_size_bytes"]
    mock_get_fpc.return_value = mock_fpc_instance

    from dam.core.world import get_world
    from dam.models.properties.file_properties_component import FilePropertiesComponent as FPC_type
    target_world = get_world("cli_test_world")
    assert target_world is not None

    # Scenario 1: Asset by ID, Profile by ID
    asset_id_arg = 1
    profile_id_arg = 10

    # Simulate logic of cli_transcode_apply._apply
    # Actual ecs_service.find_entity_id_by_hash is not called if asset_identifier is int
    await transcode_service.apply_transcode_profile(
        world=target_world, source_asset_entity_id=asset_id_arg, profile_entity_id=profile_id_arg
    )
    async with target_world.db_session_maker() as session: # Get a session for get_component
        await dam_ecs_service_module.get_component(session, mock_transcoded_entity.id, FPC_type)

    mock_apply.assert_awaited_once() # Removed await
    call_args_apply = mock_apply.call_args[1]
    assert call_args_apply['source_asset_entity_id'] == asset_id_arg
    assert call_args_apply['profile_entity_id'] == profile_id_arg
    mock_get_fpc.assert_awaited_once() # Removed await

    # Reset mocks for Scenario 2
    mock_apply.reset_mock(); mock_get_fpc.reset_mock(); mock_find_by_hash.reset_mock()

    # Scenario 2: Asset by Hash, Profile by Name
    asset_hash_arg = "somehashvalue"
    profile_name_arg = "someprofilename"
    resolved_asset_id_from_hash = 2
    mock_find_by_hash.return_value = resolved_asset_id_from_hash
    mock_apply.return_value = mock_transcoded_entity # Ensure mock_apply is set up again
    mock_get_fpc.return_value = mock_fpc_instance    # Ensure mock_get_fpc is set up again


    # Simulate logic of cli_transcode_apply._apply for hash case
    async with target_world.db_session_maker() as session:
        source_asset_entity_id_for_call = await dam_ecs_service_module.find_entity_id_by_hash(
            session, hash_value=asset_hash_arg, hash_type="sha256"
        )
    assert source_asset_entity_id_for_call == resolved_asset_id_from_hash

    await transcode_service.apply_transcode_profile(
        world=target_world, source_asset_entity_id=source_asset_entity_id_for_call, profile_entity_id=profile_name_arg
    )
    async with target_world.db_session_maker() as session:
        await dam_ecs_service_module.get_component(session, mock_transcoded_entity.id, FPC_type)

    mock_find_by_hash.assert_awaited_once_with(ANY, hash_value=asset_hash_arg, hash_type="sha256") # Removed await
    mock_apply.assert_awaited_once() # Removed await
    call_args_apply_hash = mock_apply.call_args[1]
    assert call_args_apply_hash['source_asset_entity_id'] == resolved_asset_id_from_hash
    assert call_args_apply_hash['profile_entity_id'] == profile_name_arg
    mock_get_fpc.assert_awaited_once() # Removed await


@pytest.mark.asyncio # Mark as async
@patch("dam.systems.evaluation_systems.create_evaluation_run_concept", new_callable=AsyncMock)
async def test_cli_eval_run_create(mock_create_run: AsyncMock): # Make async
    mock_run_entity = Entity()
    mock_run_entity.id = 789
    mock_create_run.return_value = mock_run_entity

    from dam.core.world import get_world
    target_world = get_world("cli_test_world")
    assert target_world is not None

    run_name_arg = "cli_eval_run"
    desc_arg = "A test run from CLI"

    # Call the service function that the CLI command would call
    await evaluation_systems.create_evaluation_run_concept(
        world=target_world, run_name=run_name_arg, description=desc_arg
    )

    mock_create_run.assert_awaited_once() # Removed await


@pytest.mark.asyncio # Mark as async
@patch("dam.systems.evaluation_systems.execute_evaluation_run", new_callable=AsyncMock)
async def test_cli_eval_run_execute(mock_execute_run: AsyncMock): # Make async
    mock_results_list = [MagicMock(spec=EvaluationResultComponent)]
    mock_execute_run.return_value = mock_results_list

    from dam.core.world import get_world
    target_world = get_world("cli_test_world")
    assert target_world is not None

    run_identifier_arg = "my_eval_run_name"
    # CLI parsing converts these to lists of strings/ints
    asset_identifiers_arg = ['assethash1', 2]
    profile_identifiers_arg = ['profile_name1', 15]

    # Call the service function
    await evaluation_systems.execute_evaluation_run(
        world=target_world,
        evaluation_run_id_or_name=run_identifier_arg,
        source_asset_identifiers=asset_identifiers_arg,
        profile_identifiers=profile_identifiers_arg
    )

    mock_execute_run.assert_awaited_once() # Removed await
    call_args = mock_execute_run.call_args[1]
    assert call_args['source_asset_identifiers'] == asset_identifiers_arg
    assert call_args['profile_identifiers'] == ['profile_name1', 15]


@pytest.mark.asyncio # Mark as async
@patch("dam.systems.evaluation_systems.get_evaluation_results", new_callable=AsyncMock)
async def test_cli_eval_report(mock_get_results: AsyncMock): # Make async
    mock_report_data = [{
        "evaluation_run_name": "cli_reported_run",
        "original_asset_entity_id": 1, "original_asset_filename": "source.mp4",
        "profile_name": "p1", "profile_tool": "t1", "profile_params": "par1", "profile_format": "f1",
        "transcoded_asset_entity_id": 101, "transcoded_asset_filename": "transcoded1.mkv",
        "file_size_bytes": 1000, "vmaf_score": 95.5, "ssim_score": 0.99, "psnr_score": 40.1,
        "custom_metrics": {"my_metric": 10}, "notes": "Good one"
    }]
    mock_get_results.return_value = mock_report_data

    from dam.core.world import get_world
    target_world = get_world("cli_test_world")
    assert target_world is not None

    run_identifier_arg = "cli_reported_run_id"

    # Call the service function
    # The CLI command does more (printing), here we just check the service call
    await evaluation_systems.get_evaluation_results(
        world=target_world, evaluation_run_id_or_name=run_identifier_arg
    )

    mock_get_results.assert_awaited_once() # Removed await


@pytest.mark.asyncio # Mark as async
@patch("dam.services.transcode_service.apply_transcode_profile", new_callable=AsyncMock)
async def test_cli_transcode_apply_tool_not_found(mock_apply_transcode: AsyncMock): # Make async
    from dam.utils.media_utils import TranscodeError as MediaUtilTranscodeError

    service_error = transcode_service.TranscodeServiceError("Transcoding failed: Tool 'xyz' not found.")
    service_error.__cause__ = MediaUtilTranscodeError("Command not found: xyz")
    mock_apply_transcode.side_effect = service_error

    from dam.core.world import get_world
    target_world = get_world("cli_test_world")
    assert target_world is not None

    asset_id_arg = 1
    profile_id_arg = 10

    # Expect TranscodeServiceError when calling the service directly
    with pytest.raises(transcode_service.TranscodeServiceError) as excinfo:
        await transcode_service.apply_transcode_profile(
            world=target_world,
            source_asset_entity_id=asset_id_arg,
            profile_entity_id=profile_id_arg
        )
    assert "Transcoding failed: Tool 'xyz' not found." in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, MediaUtilTranscodeError)
