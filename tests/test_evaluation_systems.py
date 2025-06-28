import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import json

from sqlalchemy.ext.asyncio import AsyncSession

from dam.core.world import World
from dam.models.core.entity import Entity
from dam.models.conceptual.evaluation_run_component import EvaluationRunComponent
from dam.models.conceptual.evaluation_result_component import EvaluationResultComponent
from dam.models.conceptual.transcode_profile_component import TranscodeProfileComponent
from dam.models.properties.file_properties_component import FilePropertiesComponent # For report data
from dam.models.conceptual.transcoded_variant_component import TranscodedVariantComponent # For report data


from dam.systems import evaluation_systems
from dam.services import transcode_service, ecs_service, tag_service # For setup and verification
from dam.core.exceptions import EntityNotFoundError
from dam.core.events import StartEvaluationForTranscodedAsset # Import event directly

# Import fixtures from conftest or other test files if they are reusable
# For this test file, we'll reuse some fixtures if they were defined broadly,
# or redefine specific ones. Let's assume test_world and db_session are available (e.g. from conftest or transcode_service tests)
# For simplicity, re-using the setup from test_transcode_service.py for db and world
# Note: Importing fixtures like this is fine, but specific classes like events should be from their canonical source.
from .test_transcode_service import setup_database, db_session, test_world, sample_asset_entity


@pytest.mark.asyncio
async def test_create_evaluation_run_concept(test_world: World, db_session: AsyncSession):
    run_name = "eval_run_001"
    description = "Test evaluation for AVIF quality"

    run_entity = await evaluation_systems.create_evaluation_run_concept(
        world=test_world,
        run_name=run_name,
        description=description,
        # session=db_session # Service manages its own session
    )
    assert run_entity is not None
    assert run_entity.id is not None

    async with test_world.db_session_maker() as session:
        retrieved_entity = await session.get(Entity, run_entity.id)
        assert retrieved_entity is not None

        stmt = evaluation_systems.select(EvaluationRunComponent).where(EvaluationRunComponent.entity_id == run_entity.id)
        eval_run_comp = (await session.execute(stmt)).scalars().first()

        assert eval_run_comp is not None
        assert eval_run_comp.run_name == run_name
        assert eval_run_comp.concept_name == run_name
        assert eval_run_comp.concept_description == description

        # Check for system tag
        tags = await tag_service.get_tags_for_entity(test_world, entity_id=run_entity.id, session=session)
        assert any(tag.name == "System:EvaluationRun" for tag, _ in tags)


@pytest.mark.asyncio
async def test_create_evaluation_run_already_exists(test_world: World, db_session: AsyncSession):
    run_name = "eval_run_unique_002"
    await evaluation_systems.create_evaluation_run_concept(test_world, run_name)

    with pytest.raises(evaluation_systems.EvaluationError, match=f"Evaluation run with name '{run_name}' already exists"):
        await evaluation_systems.create_evaluation_run_concept(test_world, run_name)


@pytest.mark.asyncio
async def test_get_evaluation_run_by_name_or_id(test_world: World, db_session: AsyncSession):
    run_name = "eval_run_to_get"
    created_run_entity = await evaluation_systems.create_evaluation_run_concept(test_world, run_name)

    # By name
    entity_by_name, comp_by_name = await evaluation_systems.get_evaluation_run_by_name_or_id(test_world, run_name)
    assert entity_by_name.id == created_run_entity.id
    assert comp_by_name.run_name == run_name

    # By ID
    entity_by_id, comp_by_id = await evaluation_systems.get_evaluation_run_by_name_or_id(test_world, created_run_entity.id)
    assert entity_by_id.id == created_run_entity.id
    assert comp_by_id.run_name == run_name

    with pytest.raises(evaluation_systems.EvaluationError, match="Evaluation run 'non_existent_run' not found"):
        await evaluation_systems.get_evaluation_run_by_name_or_id(test_world, "non_existent_run")
    with pytest.raises(evaluation_systems.EvaluationError, match="Evaluation run '99988' not found"):
        await evaluation_systems.get_evaluation_run_by_name_or_id(test_world, 99988)


@pytest_asyncio.fixture
async def setup_for_eval_execution(test_world: World, sample_asset_entity: tuple[int, Path]):
    source_asset_id, _ = sample_asset_entity # From test_transcode_service via import

    # Create a couple of transcode profiles
    profile1_name = "eval_prof_fast"
    profile1_entity = await transcode_service.create_transcode_profile(
        test_world, profile1_name, "ffmpeg", "-preset ultrafast {output}", "mkv"
    )
    profile2_name = "eval_prof_slow"
    profile2_entity = await transcode_service.create_transcode_profile(
        test_world, profile2_name, "ffmpeg", "-preset veryslow {output}", "mp4"
    )

    # Create an evaluation run concept
    eval_run_name = "MyVideoEvaluation"
    eval_run_entity = await evaluation_systems.create_evaluation_run_concept(
        test_world, eval_run_name, "Evaluating fast vs slow presets"
    )

    return {
        "source_asset_id": source_asset_id,
        "profile1_id": profile1_entity.id,
        "profile1_name": profile1_name,
        "profile2_id": profile2_entity.id,
        "profile2_name": profile2_name,
        "eval_run_id": eval_run_entity.id,
        "eval_run_name": eval_run_name,
    }

@pytest.mark.asyncio
@patch("dam.services.transcode_service.apply_transcode_profile", new_callable=AsyncMock) # Mock the actual transcoding call
async def test_execute_evaluation_run(
    mock_apply_transcode: AsyncMock,
    test_world: World,
    setup_for_eval_execution: dict,
    tmp_path: Path # For mocked transcoded files if needed by apply_transcode_profile mock
):
    eval_data = setup_for_eval_execution

    # Mock apply_transcode_profile to simulate successful transcoding and return a new Entity
    # It needs to create a new entity with some basic components (FPC, Hash) so that
    # execute_evaluation_run can create an EvaluationResultComponent.

    created_mock_entities_info = [] # Store (entity_id, file_size)

    async def side_effect_apply_transcode(world, source_asset_entity_id, profile_entity_id, output_parent_dir=None):
        # This mock simulates that apply_transcode_profile created a new asset.
        # It should return an Entity object.
        async with world.db_session_maker() as session:
            new_mock_entity = Entity()
            session.add(new_mock_entity)
            await session.flush() # Get ID

            # Add minimal components that execute_evaluation_run might query
            mock_filesize = 1024 * profile_entity_id # Make filesize unique per profile for test
            fpc = FilePropertiesComponent(
                entity_id=new_mock_entity.id,
                original_filename=f"transcoded_by_mock_{profile_entity_id}.dat",
                file_size_bytes=mock_filesize,
                mime_type="application/octet-stream"
            )
            session.add(fpc)
            await session.commit()
            await session.refresh(new_mock_entity)
            created_mock_entities_info.append({"id": new_mock_entity.id, "size": mock_filesize, "profile_id": profile_entity_id})
            return new_mock_entity # Return the SQLAlchemy Entity object

    mock_apply_transcode.side_effect = side_effect_apply_transcode

    results_components = await evaluation_systems.execute_evaluation_run(
        world=test_world,
        evaluation_run_id_or_name=eval_data["eval_run_id"],
        source_asset_identifiers=[eval_data["source_asset_id"]], # Use the real source asset ID
        profile_identifiers=[eval_data["profile1_id"], eval_data["profile2_name"]], # Mix ID and name
    )

    assert len(results_components) == 2 # One result for each profile applied to the source asset
    assert mock_apply_transcode.call_count == 2

    # Verify calls to apply_transcode_profile
    call_args_list = mock_apply_transcode.call_args_list

    # Check first call (profile1)
    args_p1, _ = call_args_list[0]
    assert args_p1[1] == eval_data["source_asset_id"] # source_asset_entity_id
    assert args_p1[2] == eval_data["profile1_id"]     # profile_entity_id

    # Check second call (profile2, identified by name, resolved to ID by service)
    args_p2, _ = call_args_list[1]
    assert args_p2[1] == eval_data["source_asset_id"]
    assert args_p2[2] == eval_data["profile2_id"]

    # Verify the EvaluationResultComponents created
    async with test_world.db_session_maker() as session:
        for res_comp in results_components:
            assert res_comp.id is not None
            assert res_comp.evaluation_run_entity_id == eval_data["eval_run_id"]
            assert res_comp.original_asset_entity_id == eval_data["source_asset_id"]

            # Find which mock entity this result corresponds to
            mock_info_for_result = next(info for info in created_mock_entities_info if info["id"] == res_comp.entity_id)
            assert mock_info_for_result is not None

            assert res_comp.transcoded_asset_entity_id == mock_info_for_result["id"]
            assert res_comp.transcode_profile_entity_id == mock_info_for_result["profile_id"]
            assert res_comp.file_size_bytes == mock_info_for_result["size"]
            # VMAF etc are None as we didn't mock quality calculation
            assert res_comp.vmaf_score is None


@pytest.mark.asyncio
async def test_execute_evaluation_run_no_valid_assets_or_profiles(test_world: World, setup_for_eval_execution: dict):
    eval_data = setup_for_eval_execution

    # No valid assets
    with patch("dam.services.ecs_service.find_entity_id_by_hash", AsyncMock(return_value=None)): # Make hash lookup fail
        with pytest.raises(evaluation_systems.EvaluationError, match="No valid source assets found"):
            await evaluation_systems.execute_evaluation_run(
                test_world, eval_data["eval_run_id"], ["non_existent_hash"], [eval_data["profile1_id"]]
            )

    # No valid profiles
    with pytest.raises(evaluation_systems.EvaluationError, match="No valid transcode profiles found"):
        await evaluation_systems.execute_evaluation_run(
            test_world, eval_data["eval_run_id"], [eval_data["source_asset_id"]], ["non_existent_profile_name"]
        )

@pytest.mark.asyncio
@patch("dam.services.transcode_service.apply_transcode_profile", new_callable=AsyncMock)
async def test_get_evaluation_results_formatting(
    mock_apply_transcode: AsyncMock, # Patched to allow execute_evaluation_run to run
    test_world: World,
    setup_for_eval_execution: dict,
    sample_asset_entity: tuple[int, Path] # To get original filename
):
    eval_data = setup_for_eval_execution
    _, source_asset_orig_path = sample_asset_entity # Get original path for filename

    # Simulate apply_transcode_profile as in previous test
    created_mock_entities_info = []
    async def side_effect_apply_transcode(world, source_asset_entity_id, profile_entity_id, output_parent_dir=None):
        async with world.db_session_maker() as session:
            new_mock_entity = Entity()
            session.add(new_mock_entity)
            await session.flush()

            # Determine profile name for filename
            profile_comp_stmt = evaluation_systems.select(TranscodeProfileComponent).where(TranscodeProfileComponent.entity_id == profile_entity_id)
            profile_comp_for_filename = (await session.execute(profile_comp_stmt)).scalars().first()

            fpc = FilePropertiesComponent(
                entity_id=new_mock_entity.id,
                original_filename=f"transcoded_{profile_comp_for_filename.profile_name}.dat", # Use profile name in mock filename
                file_size_bytes=12345,
                mime_type="application/octet-stream"
            )
            session.add(fpc)

            # Add TranscodedVariantComponent as it's included in eager loading options for get_evaluation_results
            tvc = TranscodedVariantComponent(entity_id=new_mock_entity.id, original_asset_entity_id=source_asset_entity_id, transcode_profile_entity_id=profile_entity_id)
            session.add(tvc)

            await session.commit()
            await session.refresh(new_mock_entity)
            created_mock_entities_info.append({"id": new_mock_entity.id, "profile_id": profile_entity_id, "filename": fpc.original_filename})
            return new_mock_entity
    mock_apply_transcode.side_effect = side_effect_apply_transcode

    # Execute the run
    await evaluation_systems.execute_evaluation_run(
        world=test_world,
        evaluation_run_id_or_name=eval_data["eval_run_name"], # Use name
        source_asset_identifiers=[eval_data["source_asset_id"]],
        profile_identifiers=[eval_data["profile1_name"], eval_data["profile2_id"]], # Mix name and ID
    )

    # Get formatted results
    formatted_results = await evaluation_systems.get_evaluation_results(
        test_world, eval_data["eval_run_name"]
    )

    assert len(formatted_results) == 2

    # Get original asset's filename for assertion
    async with test_world.db_session_maker() as session:
        orig_fpc_comp = await ecs_service.get_component_for_entity(session, eval_data["source_asset_id"], FilePropertiesComponent) # type: ignore
        expected_original_filename = orig_fpc_comp.original_filename # type: ignore

    for res_dict in formatted_results:
        assert res_dict["evaluation_run_name"] == eval_data["eval_run_name"]
        assert res_dict["original_asset_entity_id"] == eval_data["source_asset_id"]
        assert res_dict["original_asset_filename"] == expected_original_filename

        # Find corresponding mock info
        mock_info = next(info for info in created_mock_entities_info if info["id"] == res_dict["transcoded_asset_entity_id"])

        assert res_dict["transcoded_asset_filename"] == mock_info["filename"]
        assert res_dict["file_size_bytes"] == 12345 # From mock FPC

        if mock_info["profile_id"] == eval_data["profile1_id"]:
            assert res_dict["profile_name"] == eval_data["profile1_name"]
        elif mock_info["profile_id"] == eval_data["profile2_id"]:
            assert res_dict["profile_name"] == eval_data["profile2_name"]
        else:
            pytest.fail("Result profile doesn't match expectations.")

        # VMAF etc are None
        assert res_dict["vmaf_score"] is None
        assert res_dict["custom_metrics"] == {} # Empty if JSON was null/empty

    # Test getting results for a non-existent run
    with pytest.raises(evaluation_systems.EvaluationError, match="Evaluation run 'ghost_run' not found"):
        await evaluation_systems.get_evaluation_results(test_world, "ghost_run")


@pytest.mark.asyncio
async def test_execute_evaluation_run_source_asset_not_found(test_world: World, setup_for_eval_execution: dict):
    eval_data = setup_for_eval_execution

    # Mock find_entity_id_by_hash to simulate asset not found
    with patch("dam.services.ecs_service.find_entity_id_by_hash", AsyncMock(return_value=None)):
        # We expect it to log a warning and continue if other assets are valid,
        # or complete with 0 results if this was the only asset.
        results = await evaluation_systems.execute_evaluation_run(
            test_world,
            eval_data["eval_run_id"],
            source_asset_identifiers=["non_existent_hash_value"], # This hash will not be found
            profile_identifiers=[eval_data["profile1_id"]]
        )
        assert len(results) == 0 # No results should be generated if the only asset is invalid

    # Test with one valid and one invalid asset ID
    valid_asset_id = eval_data["source_asset_id"]
    invalid_asset_id = 999999 # Assume this ID does not exist

    # Mock apply_transcode_profile for the valid asset call
    with patch("dam.services.transcode_service.apply_transcode_profile", new_callable=AsyncMock) as mock_atp:
        async def side_effect_atp_valid(world, source_asset_entity_id, profile_entity_id, output_parent_dir=None):
            if source_asset_entity_id == valid_asset_id: # Only mock for the valid one
                async with world.db_session_maker() as session:
                    new_mock_entity = Entity(); session.add(new_mock_entity); await session.flush()
                    fpc = FilePropertiesComponent(entity_id=new_mock_entity.id, original_filename="valid_transcode.dat", file_size_bytes=100)
                    session.add(fpc); await session.commit(); await session.refresh(new_mock_entity)
                    return new_mock_entity
            else: # Should not be called for invalid_asset_id if get_entity_by_id check works
                raise AssertionError("apply_transcode_profile called for unexpected asset ID")
        mock_atp.side_effect = side_effect_atp_valid

        results_mixed = await evaluation_systems.execute_evaluation_run(
            test_world,
            eval_data["eval_run_id"],
            source_asset_identifiers=[valid_asset_id, invalid_asset_id],
            profile_identifiers=[eval_data["profile1_id"]]
        )
        # Should process the valid asset, skip the invalid one.
        assert len(results_mixed) == 1
        assert mock_atp.call_count == 1
        assert results_mixed[0].original_asset_entity_id == valid_asset_id
