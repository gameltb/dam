import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from typer.testing import CliRunner
from sqlalchemy.future import select
from sqlalchemy.orm import Session

from dam.cli import app
from dam.core.world import World, get_world, create_and_register_all_worlds_from_settings
from dam.core.config import Settings as AppSettings
from dam.models.core.entity import Entity
from dam.models.conceptual.evaluation_run_component import EvaluationRunComponent
from dam.models.conceptual.evaluation_result_component import EvaluationResultComponent
from dam.models.conceptual.transcode_profile_component import TranscodeProfileComponent
from dam.models.properties.file_properties_component import FilePropertiesComponent
from dam.models.conceptual.tag_concept_component import TagConceptComponent
from dam.models.conceptual.entity_tag_link_component import EntityTagLinkComponent


from dam.services import transcode_service, ecs_service as dam_ecs_service, tag_service
from dam.systems import evaluation_systems

# Assuming test_environment fixture can be imported or replicated.
from .test_cli import test_environment, click_runner, _create_dummy_file # For dummy asset creation
from .test_transcoding import _add_dummy_asset # More specific helper

# Pytest marker for async tests
pytestmark = pytest.mark.asyncio


async def _create_dummy_profile(world: World, name: str, tool: str = "mocktool", params: str = "-p", out_fmt: str = "mock") -> Entity:
    """Helper to create a dummy transcode profile."""
    return await transcode_service.create_transcode_profile(
        world=world,
        profile_name=name,
        tool_name=tool,
        parameters=params,
        output_format=out_fmt,
        description=f"Dummy profile {name}"
    )

# Test for evaluation_systems.create_evaluation_run_concept
async def test_system_create_evaluation_run_concept(test_environment):
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]

    current_test_settings = AppSettings()
    create_and_register_all_worlds_from_settings(app_settings=current_test_settings)
    target_world = get_world(default_world_name)
    assert target_world is not None

    run_name = "test_system_eval_run"
    description = "Test system evaluation run description"

    run_entity = await evaluation_systems.create_evaluation_run_concept(
        world=target_world,
        run_name=run_name,
        description=description
    )

    assert run_entity is not None
    assert run_entity.id is not None

    async with target_world.db_session_maker() as session:
        erc = await session.get(EvaluationRunComponent, run_entity.id)
        assert erc is not None
        assert erc.run_name == run_name
        assert erc.concept_name == run_name
        assert erc.concept_description == description
        assert erc.entity_id == run_entity.id # Check if entity_id is correctly set

        # Verify System:EvaluationRun tag
        tag_concept = await tag_service.get_tag_concept_by_name(session, "System:EvaluationRun")
        assert tag_concept is not None

        link_stmt = select(EntityTagLinkComponent).where(
            EntityTagLinkComponent.entity_id == run_entity.id,
            EntityTagLinkComponent.tag_concept_entity_id == tag_concept.id # type: ignore
        )
        link_result = await session.execute(link_stmt)
        link = link_result.scalar_one_or_none()
        assert link is not None, "System:EvaluationRun tag was not applied."

# Test for CLI evaluate run-create
async def test_cli_eval_run_create(test_environment, click_runner: CliRunner):
    default_world_name = test_environment["default_world_name"]

    run_name = "test_cli_eval_run"
    description = "Test CLI evaluation run description"

    result = click_runner.invoke(app, [
        "--world", default_world_name,
        "evaluate", "run-create",
        "--name", run_name,
        "--desc", description,
    ])

    assert result.exit_code == 0, f"CLI Error: {result.output}"
    assert f"Evaluation run '{run_name}'" in result.output
    assert "created successfully" in result.output

    target_world = get_world(default_world_name)
    assert target_world is not None
    async with target_world.db_session_maker() as session:
        stmt = select(EvaluationRunComponent).where(EvaluationRunComponent.run_name == run_name)
        db_run_comp = (await session.execute(stmt)).scalar_one_or_none()

        assert db_run_comp is not None
        assert db_run_comp.concept_name == run_name
        assert db_run_comp.concept_description == description

        run_entity_id = db_run_comp.entity_id
        tag_concept = await tag_service.get_tag_concept_by_name(target_world, "System:EvaluationRun", session=session)
        assert tag_concept is not None

        link_stmt = select(EntityTagLinkComponent).where(
            EntityTagLinkComponent.entity_id == run_entity_id,
            EntityTagLinkComponent.tag_concept_id == tag_concept.id
        )
        link_result = await session.execute(link_stmt)
        link = link_result.scalar_one_or_none()
        assert link is not None, "System:EvaluationRun tag was not applied via CLI."


# Test for evaluation_systems.execute_evaluation_run
async def test_system_execute_evaluation_run(test_environment, monkeypatch):
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]

    current_test_settings = AppSettings()
    create_and_register_all_worlds_from_settings(app_settings=current_test_settings)
    target_world = get_world(default_world_name)
    assert target_world is not None

    # 1. Setup: Source Asset, Transcode Profile, Evaluation Run
    source_asset = await _add_dummy_asset(target_world, tmp_path, "eval_source.txt", "evaluate this content")
    profile1 = await _create_dummy_profile(target_world, "eval_prof1")
    eval_run_concept = await evaluation_systems.create_evaluation_run_concept(target_world, "sys_exec_run")

    # 2. Mock transcode_service.apply_transcode_profile
    # It should return a mock Entity representing the transcoded asset.
    # This mock entity needs an ID and a FilePropertiesComponent for size.

    mock_transcoded_entity = Entity()
    # We need to add it to a session to get an ID, or assign one if it's transient.
    # For simplicity in mock, let's assign a distinct ID.
    mock_transcoded_entity.id = source_asset.id + 1000 # Ensure it's different

    # Create a mock FilePropertiesComponent for the mock_transcoded_entity
    mock_fpc_transcoded = FilePropertiesComponent(
        entity_id = mock_transcoded_entity.id, # Link to the mock entity
        original_filename="transcoded_eval_asset.mock",
        file_size_bytes=12345,
        mime_type="application/octet-stream"
    )

    # The service's apply_transcode_profile returns an Entity.
    # The execute_evaluation_run then fetches this entity from DB to get components.
    # So, the mock needs to ensure this entity and its FPC are "findable".
    # A simpler mock: have apply_transcode_profile return the entity that's already "in the DB" (mocked sense)
    # Or, the mock for apply_transcode_profile will also mock ecs_service.get_entity_by_id
    # and ecs_service.get_component_for_target_entity used within execute_evaluation_run.

    # Let's make the mock for `apply_transcode_profile` also handle the creation of the mock entity and its FPC
    # in a way that `execute_evaluation_run` can retrieve them.
    # This is tricky because `apply_transcode_profile` uses its own session.
    #
    # Simpler approach for this test:
    # Mock `apply_transcode_profile` to return a simple object with an `id` attribute.
    # Then, mock `ecs_service.get_entity_by_id` to return our `mock_transcoded_entity`
    # And mock `ecs_service.get_component_for_target_entity` to return `mock_fpc_transcoded`.

    mock_apply_transcode_profile = AsyncMock(return_value=MagicMock(id=mock_transcoded_entity.id)) # Returns an object with an 'id'
    monkeypatch.setattr(transcode_service, "apply_transcode_profile", mock_apply_transcode_profile)

    # When execute_evaluation_run calls ecs_service.get_entity_by_id for the transcoded asset:
    async def mock_get_entity_by_id(session, entity_id):
        if entity_id == mock_transcoded_entity.id:
            return mock_transcoded_entity # Return the actual mock Entity object
        return await dam_ecs_service.get_entity_by_id(session, entity_id) # Passthrough for others

    monkeypatch.setattr(dam_ecs_service, "get_entity_by_id", AsyncMock(side_effect=mock_get_entity_by_id))

    # When execute_evaluation_run calls ecs_service.get_component_for_target_entity:
    async def mock_get_component_for_target_entity(session, entity, component_class):
        if entity.id == mock_transcoded_entity.id and component_class == FilePropertiesComponent:
            return mock_fpc_transcoded # Return the mock FPC
        return await dam_ecs_service.get_component_for_target_entity(session, entity, component_class) # Passthrough

    monkeypatch.setattr(dam_ecs_service, "get_component_for_target_entity", AsyncMock(side_effect=mock_get_component_for_target_entity))


    # 3. Execute the evaluation run
    results = await evaluation_systems.execute_evaluation_run(
        world=target_world,
        evaluation_run_id_or_name=eval_run_concept.id,
        source_asset_identifiers=[source_asset.id],
        profile_identifiers=[profile1.id]
    )

    assert len(results) == 1
    eval_result_comp = results[0]

    assert eval_result_comp.evaluation_run_entity_id == eval_run_concept.id
    assert eval_result_comp.original_asset_entity_id == source_asset.id
    assert eval_result_comp.transcode_profile_entity_id == profile1.id
    assert eval_result_comp.transcoded_asset_entity_id == mock_transcoded_entity.id
    assert eval_result_comp.entity_id == mock_transcoded_entity.id # Attached to the transcoded entity
    assert eval_result_comp.file_size_bytes == mock_fpc_transcoded.file_size_bytes
    assert eval_result_comp.vmaf_score is None # As quality calculation is not implemented
    assert eval_result_comp.ssim_score is None
    assert eval_result_comp.psnr_score is None

    mock_apply_transcode_profile.assert_called_once_with(
        world=target_world,
        source_asset_entity_id=source_asset.id,
        profile_entity_id=profile1.id
    )

# Test for CLI evaluate run-execute
async def test_cli_eval_run_execute(test_environment, click_runner: CliRunner, monkeypatch):
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]

    current_test_settings = AppSettings()
    create_and_register_all_worlds_from_settings(app_settings=current_test_settings)
    target_world = get_world(default_world_name)
    assert target_world is not None

    # 1. Setup
    source_asset_cli = await _add_dummy_asset(target_world, tmp_path, "eval_source_cli.txt", "cli evaluate this")
    profile_cli = await _create_dummy_profile(target_world, "eval_prof_cli")
    eval_run_cli = await evaluation_systems.create_evaluation_run_concept(target_world, "cli_exec_run")

    # 2. Mock transcode_service.apply_transcode_profile (same as above system test)
    mock_transcoded_entity_cli = Entity()
    mock_transcoded_entity_cli.id = source_asset_cli.id + 2000
    mock_fpc_transcoded_cli = FilePropertiesComponent(
        entity_id=mock_transcoded_entity_cli.id,
        original_filename="transcoded_cli_asset.mock",
        file_size_bytes=56789,
        mime_type="application/octet-stream"
    )

    mock_apply_transcode_profile_cli = AsyncMock(return_value=MagicMock(id=mock_transcoded_entity_cli.id))
    monkeypatch.setattr(transcode_service, "apply_transcode_profile", mock_apply_transcode_profile_cli)

    async def mock_get_entity_by_id_cli(session, entity_id):
        if entity_id == mock_transcoded_entity_cli.id: return mock_transcoded_entity_cli
        # Fallback to actual ecs_service for other entities (like source asset, profile, run)
        return await getattr(dam_ecs_service, "get_entity_by_id_original", dam_ecs_service.get_entity_by_id)(session, entity_id)

    if not hasattr(dam_ecs_service, "get_entity_by_id_original"): # Save original if not already saved
         setattr(dam_ecs_service, "get_entity_by_id_original", dam_ecs_service.get_entity_by_id)
    monkeypatch.setattr(dam_ecs_service, "get_entity_by_id", AsyncMock(side_effect=mock_get_entity_by_id_cli))


    async def mock_get_component_for_target_entity_cli(session, entity, component_class):
        if entity.id == mock_transcoded_entity_cli.id and component_class == FilePropertiesComponent:
            return mock_fpc_transcoded_cli
        return await getattr(dam_ecs_service, "get_component_for_target_entity_original", dam_ecs_service.get_component_for_target_entity)(session, entity, component_class)

    if not hasattr(dam_ecs_service, "get_component_for_target_entity_original"):
        setattr(dam_ecs_service, "get_component_for_target_entity_original", dam_ecs_service.get_component_for_target_entity)
    monkeypatch.setattr(dam_ecs_service, "get_component_for_target_entity", AsyncMock(side_effect=mock_get_component_for_target_entity_cli))


    # 3. Execute CLI command
    result = click_runner.invoke(app, [
        "--world", default_world_name,
        "evaluate", "run-execute",
        "--run", str(eval_run_cli.id),
        "--assets", str(source_asset_cli.id),
        "--profiles", str(profile_cli.id),
    ])

    # Restore original functions if they were saved
    if hasattr(dam_ecs_service, "get_entity_by_id_original"):
        monkeypatch.setattr(dam_ecs_service, "get_entity_by_id", dam_ecs_service.get_entity_by_id_original)
    if hasattr(dam_ecs_service, "get_component_for_target_entity_original"):
        monkeypatch.setattr(dam_ecs_service, "get_component_for_target_entity", dam_ecs_service.get_component_for_target_entity_original)


    assert result.exit_code == 0, f"CLI Error: {result.output}"
    assert f"Evaluation run '{eval_run_cli.id}' completed." in result.output # CLI uses ID if ID is passed
    assert "Generated 1 results." in result.output # Based on mock setup

    # 4. Verify DB
    async with target_world.db_session_maker() as session:
        stmt = select(EvaluationResultComponent).where(
            EvaluationResultComponent.evaluation_run_entity_id == eval_run_cli.id,
            EvaluationResultComponent.original_asset_entity_id == source_asset_cli.id,
            EvaluationResultComponent.transcode_profile_entity_id == profile_cli.id
        )
        db_eval_res = (await session.execute(stmt)).scalar_one_or_none()
        assert db_eval_res is not None
        assert db_eval_res.transcoded_asset_entity_id == mock_transcoded_entity_cli.id
        assert db_eval_res.file_size_bytes == mock_fpc_transcoded_cli.file_size_bytes


# Test for CLI evaluate report
async def test_cli_eval_report(test_environment, click_runner: CliRunner, monkeypatch):
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]

    current_test_settings = AppSettings()
    create_and_register_all_worlds_from_settings(app_settings=current_test_settings)
    target_world = get_world(default_world_name)
    assert target_world is not None

    # 1. Setup: Execute a run (mocked) to generate some data
    source_asset_report = await _add_dummy_asset(target_world, tmp_path, "eval_source_report.txt", "report evaluate this")
    profile_report = await _create_dummy_profile(target_world, "eval_prof_report")
    eval_run_report_concept = await evaluation_systems.create_evaluation_run_concept(target_world, "report_exec_run_name")


    mock_transcoded_entity_report = Entity()
    mock_transcoded_entity_report.id = source_asset_report.id + 3000
    mock_fpc_transcoded_report = FilePropertiesComponent(
        entity_id=mock_transcoded_entity_report.id,
        original_filename="transcoded_report_asset.mock",
        file_size_bytes=9999,
        mime_type="application/octet-stream"
    )

    # Mock apply_transcode_profile and its downstream dependencies for execute_evaluation_run
    # Using patch context manager might be cleaner for multiple mocks if needed frequently
    mock_apply_transcode_profile_report = AsyncMock(return_value=MagicMock(id=mock_transcoded_entity_report.id))
    monkeypatch.setattr(transcode_service, "apply_transcode_profile", mock_apply_transcode_profile_report)

    async def mock_get_entity_by_id_report(session, entity_id_val):
        if entity_id_val == mock_transcoded_entity_report.id: return mock_transcoded_entity_report
        # This is a simplified mock. A real test might need to return source_asset_report etc.
        # For get_evaluation_results, it needs to fetch original asset FPC, profile name etc.
        # Let's allow passthrough for non-mocked IDs
        original_get_entity_by_id = getattr(dam_ecs_service, "get_entity_by_id_original_report", dam_ecs_service.get_entity_by_id)
        return await original_get_entity_by_id(session, entity_id_val)

    if not hasattr(dam_ecs_service, "get_entity_by_id_original_report"):
         setattr(dam_ecs_service, "get_entity_by_id_original_report", dam_ecs_service.get_entity_by_id)
    monkeypatch.setattr(dam_ecs_service, "get_entity_by_id", AsyncMock(side_effect=mock_get_entity_by_id_report))

    async def mock_get_component_for_target_entity_report(session, entity, component_class):
        if entity.id == mock_transcoded_entity_report.id and component_class == FilePropertiesComponent:
            return mock_fpc_transcoded_report
        original_get_comp = getattr(dam_ecs_service, "get_component_for_target_entity_original_report", dam_ecs_service.get_component_for_target_entity)
        return await original_get_comp(session, entity, component_class)

    if not hasattr(dam_ecs_service, "get_component_for_target_entity_original_report"):
        setattr(dam_ecs_service, "get_component_for_target_entity_original_report", dam_ecs_service.get_component_for_target_entity)
    monkeypatch.setattr(dam_ecs_service, "get_component_for_target_entity", AsyncMock(side_effect=mock_get_component_for_target_entity_report))


    await evaluation_systems.execute_evaluation_run(
        world=target_world,
        evaluation_run_id_or_name=eval_run_report_concept.id, # Use ID
        source_asset_identifiers=[source_asset_report.id],
        profile_identifiers=[profile_report.id]
    )

    # 2. Call CLI report command (by name of run)
    result = click_runner.invoke(app, [
        "--world", default_world_name,
        "evaluate", "report",
        "--run", eval_run_report_concept.run_name, # Use name for CLI
    ])

    # Restore original functions
    if hasattr(dam_ecs_service, "get_entity_by_id_original_report"):
        monkeypatch.setattr(dam_ecs_service, "get_entity_by_id", dam_ecs_service.get_entity_by_id_original_report)
    if hasattr(dam_ecs_service, "get_component_for_target_entity_original_report"):
        monkeypatch.setattr(dam_ecs_service, "get_component_for_target_entity", dam_ecs_service.get_component_for_target_entity_original_report)

    assert result.exit_code == 0, f"CLI Error: {result.output}"

    # Check for key pieces of information in the report output
    assert f"--- Evaluation Report for Run: '{eval_run_report_concept.run_name}' ---" in result.output
    assert f"Original Asset: {source_asset_report.get_component(FilePropertiesComponent).original_filename}" in result.output # type: ignore
    assert f"Profile: {profile_report.get_component(TranscodeProfileComponent).profile_name}" in result.output # type: ignore
    assert f"Transcoded Asset: {mock_fpc_transcoded_report.original_filename}" in result.output
    assert f"File Size: {mock_fpc_transcoded_report.file_size_bytes} bytes" in result.output
    assert "VMAF: N/A" in result.output # Since it's not calculated
