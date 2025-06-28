import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from dam.core.config import Settings, WorldConfig
from dam.core.database import DatabaseManager
# Corrected event imports:
from dam.core.events import AssetFileIngestionRequested
from dam.services.transcode_service import (
    TranscodeJobRequested,
    TranscodeJobCompleted,
    TranscodeJobFailed,
    StartEvaluationForTranscodedAsset # This is also defined in transcode_service.py
)
from dam.core.world import World, clear_world_registry
from dam.core.world_setup import initialize_world_resources, register_core_systems
from dam.models import Base # Make sure Base is imported for metadata operations
from dam.models.conceptual import EvaluationResultComponent, EvaluationRunComponent, TranscodedVariantComponent, TranscodeProfileComponent
from dam.models.core import Entity, FileLocationComponent
from dam.models.properties import FilePropertiesComponent
# Import transcode_service module itself, not individual names if already covered by the module import
from dam.services import ecs_service, transcode_service
# Specific items from transcode_service are still needed if not prefixed with transcode_service.
from dam.services.transcode_service import (
    TranscodeJob,
    TranscodeManager,
    TranscodingError,
    check_transcode_job_status,
    execute_transcode_job_task,
    get_transcode_job_from_db,
    handle_transcode_job_completed,
    handle_transcode_job_failed,
    handle_transcode_job_request,
    select_transcode_profile_for_asset
    # create_evaluation_run_concept, # These are now in transcode_service but test might call them directly
    # link_evaluation_result_to_run,
    # get_evaluation_results_for_run
)
from dam.systems.evaluation_systems import evaluate_transcode_output

test_settings_dict = {
    "DAM_WORLDS_CONFIG": """
    {
        "test_world": {
            "DATABASE_URL": "sqlite+aiosqlite:///./test_transcode_service.db",
            "ASSET_STORAGE_PATH": "./test_assets_transcode",
            "TRANSCODE_PROFILES": [
                {
                    "name": "test_profile_heic_to_jpeg", "description": "Test HEIC to JPEG",
                    "input_file_types": [".heic"], "output_file_type": ".jpg",
                    "output_options": {"format": "jpeg", "quality": 85},
                    "command_template": "convert {input_path} -quality {quality} {output_path}"
                },
                {
                    "name": "test_profile_any_to_png", "description": "Test Any to PNG",
                    "input_file_types": ["*"], "output_file_type": ".png",
                    "output_options": {"format": "png"},
                    "command_template": "convert {input_path} {output_path}"
                }
            ]
        }
    }
    """,
    "DAM_DEFAULT_WORLD_NAME": "test_world", "TESTING_MODE": True,
    "TRANSCODING_TEMP_DIR": "./test_transcode_temp", "TRANSCODING_MAX_CONCURRENT_JOBS": 1,
}
test_settings = Settings(**test_settings_dict)

@pytest_asyncio.fixture(scope="module", autouse=True)
def event_loop_module():
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); yield loop; loop.close()

@pytest_asyncio.fixture
async def test_world_env():
    clear_world_registry()
    storage_path = Path(test_settings.get_world_config("test_world").ASSET_STORAGE_PATH)
    transcode_temp_path = Path(test_settings.TRANSCODING_TEMP_DIR)
    if storage_path.exists(): shutil.rmtree(storage_path)
    if transcode_temp_path.exists(): shutil.rmtree(transcode_temp_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    transcode_temp_path.mkdir(parents=True, exist_ok=True)
    db_file_path_str = test_settings.get_world_config("test_world").DATABASE_URL.replace("sqlite+aiosqlite:///", "")
    db_file = Path(db_file_path_str)
    if db_file.exists(): os.remove(db_file)

    world_config = test_settings.get_world_config("test_world")
    world = World(world_config=world_config)

    with patch('dam.core.world_setup.global_app_settings', test_settings), \
         patch('dam.core.world.global_app_settings', test_settings):
        initialize_world_resources(world)
        register_core_systems(world)
        db_mngr = world.get_resource(DatabaseManager)
        assert db_mngr is not None
        async with db_mngr.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        yield world
        async with db_mngr.engine.begin() as conn: # Teardown
            await conn.run_sync(Base.metadata.drop_all)
        await db_mngr.engine.dispose()

    clear_world_registry()
    if db_file.exists(): os.remove(db_file)
    if storage_path.exists(): shutil.rmtree(storage_path)
    if transcode_temp_path.exists(): shutil.rmtree(transcode_temp_path)

@pytest_asyncio.fixture
async def db_session(test_world_env: World) -> AsyncSession:
    db_mngr = test_world_env.get_resource(DatabaseManager)
    async with db_mngr.session_local() as session:
        yield session

@pytest_asyncio.fixture
async def transcode_manager(test_world_env: World, event_loop_module: asyncio.AbstractEventLoop):
    manager = TranscodeManager(
        world=test_world_env,
        max_concurrent_jobs=test_settings.TRANSCODING_MAX_CONCURRENT_JOBS,
        temp_dir=test_settings.TRANSCODING_TEMP_DIR
    )
    yield manager
    # Simplified cleanup, actual manager might need a stop() method
    if hasattr(manager, '_job_queue') and not manager._job_queue.empty():
        while not manager._job_queue.empty(): manager._job_queue.get_nowait(); manager._job_queue.task_done()
    if hasattr(manager, '_active_jobs'): manager._active_jobs.clear()

@pytest.mark.asyncio
async def test_create_transcode_profile(db_session: AsyncSession):
    profile = await transcode_service.create_transcode_profile(
        session=db_session, name="jpeg_to_png_test", description="Convert JPEG to PNG",
        input_file_types=[".jpg", ".jpeg"], output_file_type=".png",
        command_template="convert {input_path} {output_path}", output_options={"compression": 6}
    )
    await db_session.commit()
    assert profile.id is not None
    retrieved = await db_session.get(TranscodeProfileComponent, profile.id)
    assert retrieved is not None and retrieved.name == "jpeg_to_png_test"

@pytest.mark.asyncio
async def test_handle_transcode_job_request(db_session: AsyncSession, transcode_manager: TranscodeManager, test_world_env: World):
    source_asset = await ecs_service.create_entity(db_session)
    world_config = test_world_env.get_resource(WorldConfig)
    profile_config = next(p for p in world_config.TRANSCODE_PROFILES if p.name == "test_profile_any_to_png")
    profile_db = await transcode_service.create_transcode_profile(
        db_session, profile_config.name, profile_config.description, profile_config.input_file_types,
        profile_config.output_file_type, profile_config.command_template, profile_config.output_options
    )
    await db_session.commit()
    event = TranscodeJobRequested( # Uses corrected import
        world_name=test_world_env.name, source_entity_id=source_asset.id, profile_id=profile_db.id, priority=5
    )
    transcode_manager.add_job_to_queue = AsyncMock()
    await handle_transcode_job_request(event, world=test_world_env, manager=transcode_manager, session=db_session)
    stmt = select(transcode_service.TranscodeJobDB).where(
        transcode_service.TranscodeJobDB.source_entity_id == source_asset.id)
    created_job_db = (await db_session.execute(stmt)).scalars().first()
    assert created_job_db is not None
    transcode_manager.add_job_to_queue.assert_called_once()

@patch('asyncio.create_subprocess_shell')
@pytest.mark.asyncio
async def test_execute_transcode_job_task_success(mock_shell, db_session: AsyncSession, test_world_env: World, transcode_manager: TranscodeManager):
    source_asset = await ecs_service.create_entity(db_session)
    fname = "test_input.heic"; world_cfg = test_world_env.get_resource(WorldConfig)
    fpath = Path(world_cfg.ASSET_STORAGE_PATH) / fname; fpath.touch()
    await ecs_service.add_component(db_session, source_asset.id, FileLocationComponent(full_path=str(fpath), file_name=fname))
    profile_cfg = next(p for p in world_cfg.TRANSCODE_PROFILES if p.name == "test_profile_heic_to_jpeg")
    profile_db = await transcode_service.create_transcode_profile(
        db_session, profile_cfg.name, profile_cfg.description, profile_cfg.input_file_types,
        profile_cfg.output_file_type, profile_cfg.command_template, profile_cfg.output_options)
    job_db = await transcode_service.create_transcode_job_in_db(db_session, test_world_env.name, source_asset.id, profile_db.id)
    await db_session.commit()

    mock_proc = AsyncMock(); mock_proc.returncode = 0; mock_proc.communicate.return_value = (b"",b"")
    mock_shell.return_value = mock_proc
    job = TranscodeJob(job_id=job_db.id, world_name=test_world_env.name, source_entity_id=source_asset.id, profile_id=profile_db.id, priority=job_db.priority)
    test_world_env.dispatch_event = AsyncMock()

    await execute_transcode_job_task(job, test_world_env, transcode_manager.temp_dir, db_session)

    updated_job = await get_transcode_job_from_db(db_session, job_db.id)
    assert updated_job is not None and updated_job.status == "COMPLETED"
    test_world_env.dispatch_event.assert_called_once()
    dispatched_event = test_world_env.dispatch_event.call_args[0][0]
    assert isinstance(dispatched_event, TranscodeJobCompleted) # Uses corrected import

@patch('asyncio.create_subprocess_shell')
@pytest.mark.asyncio
async def test_execute_transcode_job_task_failure(mock_shell, db_session: AsyncSession, test_world_env: World, transcode_manager: TranscodeManager):
    source_asset = await ecs_service.create_entity(db_session)
    fpath = Path(test_world_env.get_resource(WorldConfig).ASSET_STORAGE_PATH) / "test_fail.bad"; fpath.touch()
    await ecs_service.add_component(db_session, source_asset.id, FileLocationComponent(full_path=str(fpath), file_name="test_fail.bad"))
    profile_db = await transcode_service.create_transcode_profile(db_session, "fail_profile_exec", "d", [".in"],".out", "failing_cmd")
    job_db = await transcode_service.create_transcode_job_in_db(db_session, test_world_env.name, source_asset.id, profile_db.id)
    await db_session.commit()

    mock_proc = AsyncMock(); mock_proc.returncode = 1; mock_proc.communicate.return_value = (b"", b"Error")
    mock_shell.return_value = mock_proc
    job = TranscodeJob(job_id=job_db.id, world_name=test_world_env.name, source_entity_id=source_asset.id, profile_id=profile_db.id, priority=1)
    test_world_env.dispatch_event = AsyncMock()

    await execute_transcode_job_task(job, test_world_env, transcode_manager.temp_dir, db_session)
    updated_job = await get_transcode_job_from_db(db_session, job_db.id)
    assert updated_job is not None and updated_job.status == "FAILED"
    dispatched_event = test_world_env.dispatch_event.call_args[0][0]
    assert isinstance(dispatched_event, TranscodeJobFailed) # Uses corrected import

@pytest.mark.asyncio
async def test_handle_transcode_job_completed(db_session: AsyncSession, test_world_env: World):
    source_asset = await ecs_service.create_entity(db_session); output_asset = await ecs_service.create_entity(db_session)
    profile = await transcode_service.create_transcode_profile(db_session, "handler_profile_comp", "D", [".in"], ".out", "cmd")
    job_db = await transcode_service.create_transcode_job_in_db(db_session, test_world_env.name, source_asset.id, profile.id)
    job_db.status = "PROCESSING"; job_db.output_entity_id = output_asset.id # type: ignore
    db_session.add(job_db); await db_session.commit()

    event = TranscodeJobCompleted(job_id=job_db.id, world_name=test_world_env.name, source_entity_id=source_asset.id,
                                  profile_id=profile.id, output_entity_id=output_asset.id, output_file_path="/path/to/out") # Uses corrected import
    test_world_env.dispatch_event = AsyncMock()
    await handle_transcode_job_completed(event, world=test_world_env, session=db_session)

    tv_comp = await ecs_service.get_component(db_session, output_asset.id, TranscodedVariantComponent)
    assert tv_comp is not None
    ingestion_event = test_world_env.dispatch_event.call_args[0][0]
    assert isinstance(ingestion_event, AssetFileIngestionRequested)

@pytest.mark.asyncio
async def test_handle_transcode_job_failed(db_session: AsyncSession, test_world_env: World):
    source_asset = await ecs_service.create_entity(db_session)
    profile = await transcode_service.create_transcode_profile(db_session, "fail_handler_profile_final", "D", [".in"], ".out", "cmd")
    job_db = await transcode_service.create_transcode_job_in_db(db_session, test_world_env.name, source_asset.id, profile.id)
    job_db.status = "PROCESSING"; db_session.add(job_db); await db_session.commit() # type: ignore

    event = TranscodeJobFailed(job_id=job_db.id, world_name=test_world_env.name, source_entity_id=source_asset.id,
                               profile_id=profile.id, error_message="Test failure") # Uses corrected import
    await handle_transcode_job_failed(event, world=test_world_env, session=db_session)
    final_job_db = await get_transcode_job_from_db(db_session, job_db.id)
    assert final_job_db is not None and final_job_db.status == "FAILED"

# Simplified remaining tests
@pytest.mark.asyncio
async def test_check_transcode_job_status_logic(db_session: AsyncSession, test_world_env: World):
    source_asset = await ecs_service.create_entity(db_session)
    await ecs_service.add_component(db_session, source_asset.id, FileLocationComponent(full_path="dummy.in", file_name="dummy.in"))
    profile = await transcode_service.create_transcode_profile(db_session, "status_profile_full", "D", [".in"], ".out", "cmd")
    await db_session.commit()
    job_db = await transcode_service.create_transcode_job_in_db(db_session, test_world_env.name, source_asset.id, profile.id)
    await db_session.commit()
    assert await check_transcode_job_status(db_session, job_db.id, test_world_env) is True

@pytest.mark.asyncio
async def test_evaluation_system_flow(db_session: AsyncSession, test_world_env: World):
    eval_run = await transcode_service.create_evaluation_run_concept(db_session, "EvalSysRunFull", "Test")
    source = await ecs_service.create_entity(db_session); await ecs_service.add_component(db_session, source.id, FileLocationComponent(full_path="s.txt", file_name="s.txt"))
    transcoded = await ecs_service.create_entity(db_session); await ecs_service.add_component(db_session, transcoded.id, FileLocationComponent(full_path="t.txt", file_name="t.txt"))
    profile = await transcode_service.create_transcode_profile(db_session, "eval_sys_p_full", "D", [".txt"],".txt", "cmd", {"eval_func": "dummy_eval"})
    await ecs_service.add_component(db_session, transcoded.id, TranscodedVariantComponent(source_entity_id=source.id, transcode_profile_id=profile.id))
    await db_session.commit()

    event = StartEvaluationForTranscodedAsset(world_name=test_world_env.name, evaluation_run_id=eval_run.id, transcoded_asset_id=transcoded.id) # Uses corrected import
    async def dummy_eval_func(src_path, trans_path, opts): return {"metric":1}
    with patch('dam.systems.evaluation_systems.EVALUATION_FUNCTION_REGISTRY', {"dummy_eval": dummy_eval_func}):
        await evaluate_transcode_output(event, world=test_world_env, session=db_session)
    await db_session.commit()
    results = await transcode_service.get_evaluation_results_for_run(db_session, eval_run.id) # type: ignore
    assert len(results) == 1
    assert results[0].transcoded_asset_id == transcoded.id # type: ignore
