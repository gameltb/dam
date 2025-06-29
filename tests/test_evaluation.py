from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.future import select

from dam.core.config import Settings as AppSettings
from dam.core.world import World, create_and_register_all_worlds_from_settings, get_world
from dam.models.conceptual.entity_tag_link_component import EntityTagLinkComponent
from dam.models.conceptual.evaluation_result_component import EvaluationResultComponent
from dam.models.conceptual.evaluation_run_component import EvaluationRunComponent
from dam.models.conceptual.transcode_profile_component import TranscodeProfileComponent
from dam.models.core.entity import Entity
from dam.models.properties.file_properties_component import FilePropertiesComponent
from dam.services import ecs_service as dam_ecs_service
from dam.services import tag_service, transcode_service
from dam.systems import evaluation_systems

from .test_cli import (
    test_environment,  # noqa: F401
)

# Assuming test_environment fixture can be imported or replicated.
from .test_transcoding import _add_dummy_asset  # More specific helper

# Pytest marker for async tests
pytestmark = pytest.mark.asyncio


async def _create_dummy_profile(
    world: World, name: str, tool: str = "mocktool", params: str = "-p {input} {output}", out_fmt: str = "mock"
) -> Entity:
    """Helper to create a dummy transcode profile."""
    return await transcode_service.create_transcode_profile(
        world=world,
        profile_name=name,
        tool_name=tool,
        parameters=params,
        output_format=out_fmt,
        description=f"Dummy profile {name}",
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
        world=target_world, run_name=run_name, description=description
    )

    assert run_entity is not None
    assert run_entity.id is not None

    async with target_world.db_session_maker() as session:
        erc = await session.get(EvaluationRunComponent, run_entity.id)
        assert erc is not None
        assert erc.run_name == run_name
        assert erc.concept_name == run_name
        assert erc.concept_description == description
        assert erc.entity_id == run_entity.id  # Check if entity_id is correctly set

        # Verify System:EvaluationRun tag
        tag_concept = await tag_service.get_tag_concept_by_name(session, "System:EvaluationRun")
        assert tag_concept is not None

        link_stmt = select(EntityTagLinkComponent).where(
            EntityTagLinkComponent.entity_id == run_entity.id,
            EntityTagLinkComponent.tag_concept_entity_id == tag_concept.id,  # type: ignore
        )
        link_result = await session.execute(link_stmt)
        link = link_result.scalar_one_or_none()
        assert link is not None, "System:EvaluationRun tag was not applied."


# Test for CLI evaluate run-create (Refactored to call system directly)
async def test_cli_eval_run_create(test_environment):  # Removed click_runner
    default_world_name = test_environment["default_world_name"]
    tmp_path = test_environment["tmp_path"]  # Added tmp_path for consistency if needed, though not used here

    # Ensure worlds are properly initialized for the test context
    current_test_settings = AppSettings()
    create_and_register_all_worlds_from_settings(app_settings=current_test_settings)
    target_world = get_world(default_world_name)
    assert target_world is not None

    run_name = "test_cli_eval_run"
    description = "Test CLI evaluation run description"

    # Directly call the system function, similar to test_system_create_evaluation_run_concept
    run_entity = await evaluation_systems.create_evaluation_run_concept(
        world=target_world, run_name=run_name, description=description
    )
    assert run_entity is not None, "Evaluation run entity was not created."

    # Verify the component and tag in the database
    async with target_world.db_session_maker() as session:
        stmt = select(EvaluationRunComponent).where(EvaluationRunComponent.run_name == run_name)
        db_run_comp = (await session.execute(stmt)).scalar_one_or_none()

        assert db_run_comp is not None, "EvaluationRunComponent not found in DB"
        assert db_run_comp.concept_name == run_name
        assert db_run_comp.concept_description == description
        assert db_run_comp.entity_id == run_entity.id  # Check entity_id linkage

        # Verify System:EvaluationRun tag
        # The tag_service.get_tag_concept_by_name now expects session as its first argument
        # if world is not provided, or world and session if both are needed.
        # The create_evaluation_run_concept internally calls tag_service correctly.
        tag_concept = await tag_service.get_tag_concept_by_name(session, "System:EvaluationRun")  # Pass session
        assert tag_concept is not None, "System:EvaluationRun tag concept not found."

        link_stmt = select(EntityTagLinkComponent).where(
            EntityTagLinkComponent.entity_id == run_entity.id,
            EntityTagLinkComponent.tag_concept_entity_id == tag_concept.id,  # Use tag_concept_entity_id
        )
        link_result = await session.execute(link_stmt)
        link = link_result.scalar_one_or_none()
        assert link is not None, "System:EvaluationRun tag was not applied to the run entity."


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
    mock_transcoded_entity.id = source_asset.id + 1000  # Ensure it's different

    # Create a mock FilePropertiesComponent for the mock_transcoded_entity
    # Use MagicMock instead of instantiating the actual component directly
    mock_fpc_transcoded = MagicMock(spec=FilePropertiesComponent)
    mock_fpc_transcoded.entity_id = mock_transcoded_entity.id
    mock_fpc_transcoded.original_filename = "transcoded_eval_asset.mock"
    mock_fpc_transcoded.file_size_bytes = 12345
    mock_fpc_transcoded.mime_type = "application/octet-stream"
    # Ensure it has an 'id' attribute if any code tries to access component.id (usually PK of component table)
    # For a secondary component, this might be different from entity_id.
    # Let's assume it might need an id, perhaps same as entity_id for simplicity in mock.
    mock_fpc_transcoded.id = mock_transcoded_entity.id  # Or some other mock ID if needed

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

    mock_apply_transcode_profile = AsyncMock(
        return_value=MagicMock(id=mock_transcoded_entity.id)
    )  # Returns an object with an 'id'
    monkeypatch.setattr(transcode_service, "apply_transcode_profile", mock_apply_transcode_profile)

    # When execute_evaluation_run calls ecs_service.get_entity for the transcoded asset:
    # Store true original functions before applying mocks
    true_original_get_entity = dam_ecs_service.get_entity
    true_original_get_component = dam_ecs_service.get_component

    # When execute_evaluation_run calls ecs_service.get_entity for the transcoded asset:
    async def mock_get_entity_side_effect(session, entity_id):
        if entity_id == mock_transcoded_entity.id:
            return mock_transcoded_entity  # Return the actual mock Entity object
        # Passthrough for other entity IDs using the true original function
        return await true_original_get_entity(session, entity_id)

    monkeypatch.setattr(dam_ecs_service, "get_entity", AsyncMock(side_effect=mock_get_entity_side_effect))

    # When execute_evaluation_run calls ecs_service.get_component:
    async def mock_get_component_side_effect(session, entity_id, component_class):
        if entity_id == mock_transcoded_entity.id and component_class == FilePropertiesComponent:
            return mock_fpc_transcoded  # Return the mock FPC
        # Passthrough for other calls using the true original function
        return await true_original_get_component(session, entity_id, component_class)

    monkeypatch.setattr(dam_ecs_service, "get_component", AsyncMock(side_effect=mock_get_component_side_effect))

    # Monkeypatch will automatically restore the original functions at teardown.
    # No need for manual _original_ attributes or restoration.

    # 3. Execute the evaluation run
    results = await evaluation_systems.execute_evaluation_run(
        world=target_world,
        evaluation_run_id_or_name=eval_run_concept.id,
        source_asset_identifiers=[source_asset.id],
        profile_identifiers=[profile1.id],
    )

    assert len(results) == 1
    eval_result_comp = results[0]

    assert eval_result_comp.evaluation_run_entity_id == eval_run_concept.id
    assert eval_result_comp.original_asset_entity_id == source_asset.id
    assert eval_result_comp.transcode_profile_entity_id == profile1.id
    assert eval_result_comp.transcoded_asset_entity_id == mock_transcoded_entity.id
    assert eval_result_comp.entity_id == mock_transcoded_entity.id  # Attached to the transcoded entity
    assert eval_result_comp.file_size_bytes == mock_fpc_transcoded.file_size_bytes
    assert eval_result_comp.vmaf_score is None  # As quality calculation is not implemented
    assert eval_result_comp.ssim_score is None
    assert eval_result_comp.psnr_score is None

    mock_apply_transcode_profile.assert_called_once_with(
        world=target_world, source_asset_entity_id=source_asset.id, profile_entity_id=profile1.id
    )


# Test for CLI evaluate run-execute (Refactored to call system directly)
async def test_cli_eval_run_execute(test_environment, monkeypatch):  # Removed click_runner
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

    # 2. Mock transcode_service.apply_transcode_profile (same as test_system_execute_evaluation_run)
    mock_transcoded_entity_cli = Entity()
    # Assign a unique ID. Ensure it's different from source_asset_cli.id, profile_cli.id, eval_run_cli.id
    # A robust way is to use IDs that are unlikely to clash or manage a counter.
    # For this test, simple addition should be fine if IDs are small integers.
    mock_transcoded_entity_cli.id = source_asset_cli.id + 2000
    # Use MagicMock for the FilePropertiesComponent
    mock_fpc_transcoded_cli = MagicMock(spec=FilePropertiesComponent)
    mock_fpc_transcoded_cli.entity_id = mock_transcoded_entity_cli.id
    mock_fpc_transcoded_cli.original_filename = "transcoded_cli_asset.mock"
    mock_fpc_transcoded_cli.file_size_bytes = 56789
    mock_fpc_transcoded_cli.mime_type = "application/octet-stream"
    mock_fpc_transcoded_cli.id = mock_transcoded_entity_cli.id  # Mock component's own PK

    mock_apply_transcode_profile_cli = AsyncMock(return_value=MagicMock(id=mock_transcoded_entity_cli.id))
    monkeypatch.setattr(transcode_service, "apply_transcode_profile", mock_apply_transcode_profile_cli)

    # Mock ecs_service.get_entity_by_id and ecs_service.get_component_for_target_entity
    # to allow execute_evaluation_run to find the mocked transcoded asset and its FPC.
    # Store original functions to restore them, preventing interference with other tests.
    # Store original functions to restore them, preventing interference with other tests.
    original_get_entity = dam_ecs_service.get_entity
    original_get_component = dam_ecs_service.get_component  # Correct function name

    async def mock_get_entity_cli_direct(session, entity_id_val):
        if entity_id_val == mock_transcoded_entity_cli.id:
            return mock_transcoded_entity_cli
        # Important: Fallback to the original function for other entity IDs
        return await original_get_entity(session, entity_id_val)

    monkeypatch.setattr(dam_ecs_service, "get_entity", AsyncMock(side_effect=mock_get_entity_cli_direct))

    async def mock_get_component_cli_direct(session, entity_id, component_class):  # Changed signature
        if entity_id == mock_transcoded_entity_cli.id and component_class == FilePropertiesComponent:
            return mock_fpc_transcoded_cli
        return await original_get_component(session, entity_id, component_class)  # Call original correct function

    monkeypatch.setattr(
        dam_ecs_service, "get_component", AsyncMock(side_effect=mock_get_component_cli_direct)
    )  # Patch correct function

    # 3. Execute the system function directly
    results = await evaluation_systems.execute_evaluation_run(
        world=target_world,
        evaluation_run_id_or_name=eval_run_cli.id,  # Use ID of the run concept
        source_asset_identifiers=[source_asset_cli.id],  # Use ID of source asset
        profile_identifiers=[profile_cli.id],  # Use ID of profile
    )

    # Restore original functions
    monkeypatch.setattr(dam_ecs_service, "get_entity", original_get_entity)
    monkeypatch.setattr(dam_ecs_service, "get_component", original_get_component)  # Restore correct function

    # Assertions on results (no CLI output to check)
    assert len(results) == 1, "Expected one result from evaluation run."
    # Further assertions can be made on the content of 'results' if needed,
    # similar to test_system_execute_evaluation_run.

    # 4. Verify DB (same as original test)
    async with target_world.db_session_maker() as session:
        stmt = select(EvaluationResultComponent).where(
            EvaluationResultComponent.evaluation_run_entity_id == eval_run_cli.id,
            EvaluationResultComponent.original_asset_entity_id == source_asset_cli.id,
            EvaluationResultComponent.transcode_profile_entity_id == profile_cli.id,
        )
        db_eval_res = (await session.execute(stmt)).scalar_one_or_none()
        assert db_eval_res is not None, "EvaluationResultComponent not found in DB"
        assert db_eval_res.transcoded_asset_entity_id == mock_transcoded_entity_cli.id
        assert db_eval_res.file_size_bytes == mock_fpc_transcoded_cli.file_size_bytes


# Test for CLI evaluate report (Refactored to call system directly)
async def test_cli_eval_report(test_environment, monkeypatch):  # Removed click_runner
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]

    current_test_settings = AppSettings()
    create_and_register_all_worlds_from_settings(app_settings=current_test_settings)
    target_world = get_world(default_world_name)
    assert target_world is not None

    # 1. Setup: Execute a run (mocked) to generate some data
    source_asset_report = await _add_dummy_asset(
        target_world, tmp_path, "eval_source_report.txt", "report evaluate this"
    )
    profile_report = await _create_dummy_profile(target_world, "eval_prof_report")
    eval_run_report_concept = await evaluation_systems.create_evaluation_run_concept(
        target_world, "report_exec_run_name"
    )

    # Create a real placeholder entity for the "transcoded" asset
    async with target_world.db_session_maker() as session:
        real_mock_transcoded_entity = Entity()
        session.add(real_mock_transcoded_entity)
        await session.commit()  # Commit to get an ID
        await session.refresh(real_mock_transcoded_entity)
        # Create and add a FilePropertiesComponent for this mock transcoded entity
        # This FPC will be fetched by execute_evaluation_run
        mock_fpc_transcoded_report = FilePropertiesComponent(
            original_filename="transcoded_report_asset.mock", file_size_bytes=9999, mime_type="application/octet-stream"
        )
        await dam_ecs_service.add_component_to_entity(
            session, real_mock_transcoded_entity.id, mock_fpc_transcoded_report
        )
        await session.commit()

    # Mock apply_transcode_profile to return the ID of this real entity
    mock_apply_transcode_profile_report = AsyncMock(return_value=MagicMock(id=real_mock_transcoded_entity.id))
    monkeypatch.setattr(transcode_service, "apply_transcode_profile", mock_apply_transcode_profile_report)

    # Store original ecs_service functions that might be restored
    original_get_entity = dam_ecs_service.get_entity
    original_get_component = dam_ecs_service.get_component

    # No need to mock get_entity for the real_mock_transcoded_entity as it's real.
    # We might still need to mock get_component if execute_evaluation_run needs other components
    # that _add_dummy_asset or _create_dummy_profile don't create, but for FPC it should be fine now.
    # The test only seems to rely on FPC for the transcoded asset.

    # Execute the run to populate data for the report
    await evaluation_systems.execute_evaluation_run(
        world=target_world,
        evaluation_run_id_or_name=eval_run_report_concept.id,
        source_asset_identifiers=[source_asset_report.id],
        profile_identifiers=[profile_report.id],
    )

    # Restore ecs_service functions BEFORE calling get_evaluation_results,
    # as get_evaluation_results uses these services internally without our mocks.
    monkeypatch.setattr(dam_ecs_service, "get_entity", original_get_entity)
    monkeypatch.setattr(dam_ecs_service, "get_component", original_get_component)  # Restore correct function

    # Fetch the component to get its name
    async with target_world.db_session_maker() as session:
        eval_run_comp_for_report = await session.get(EvaluationRunComponent, eval_run_report_concept.id)
        assert eval_run_comp_for_report is not None
        evaluation_run_name_for_report = eval_run_comp_for_report.run_name

    # 2. Call the system function for generating report data
    report_data = await evaluation_systems.get_evaluation_results(
        world=target_world, evaluation_run_id_or_name=evaluation_run_name_for_report
    )

    # Assertions on the structure and content of report_data
    assert report_data is not None
    assert len(report_data) == 1, "Expected one result in the report data"

    result_item = report_data[0]
    assert result_item["evaluation_run_name"] == evaluation_run_name_for_report
    # Verify original asset details (FilePropertiesComponent for original asset is fetched by get_evaluation_results)
    # We need to ensure the original asset (source_asset_report) and its FPC are in the DB correctly.
    # _add_dummy_asset should have handled this.
    source_fpc = await dam_ecs_service.get_component(
        target_world.db_session_maker(), source_asset_report.id, FilePropertiesComponent
    )  # type: ignore
    assert source_fpc is not None
    assert result_item["original_asset_filename"] == source_fpc.original_filename  # type: ignore

    # Verify profile details (TranscodeProfileComponent for profile_report is fetched by get_evaluation_results)
    profile_comp = await dam_ecs_service.get_component(
        target_world.db_session_maker(), profile_report.id, TranscodeProfileComponent
    )  # type: ignore
    assert profile_comp is not None
    assert result_item["profile_name"] == profile_comp.profile_name  # type: ignore

    assert result_item["transcoded_asset_filename"] == mock_fpc_transcoded_report.original_filename
    assert result_item["file_size_bytes"] == mock_fpc_transcoded_report.file_size_bytes
    assert result_item["vmaf_score"] is None  # Placeholder value
    # Add more assertions on other fields in result_item as necessary
