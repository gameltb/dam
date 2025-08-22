from pathlib import Path
from unittest.mock import AsyncMock, MagicMock  # Added MagicMock

import pytest
import typer  # Added import
from sqlalchemy.future import select

from dam import cli as dam_cli  # Import cli with an alias
from dam.core.config import Settings as AppSettings
from dam.core.events import AssetFileIngestionRequested
from dam.core.stages import SystemStage
from dam.core.world import World, create_and_register_all_worlds_from_settings, get_world
from dam.models.conceptual.transcode_profile_component import TranscodeProfileComponent
from dam.models.conceptual.transcoded_variant_component import TranscodedVariantComponent
from dam.models.core.entity import Entity
from dam.models.core.file_location_component import FileLocationComponent
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam.models.properties.file_properties_component import FilePropertiesComponent
from dam.models.tags import EntityTagLinkComponent  # Updated imports
from dam.services import ecs_service as dam_ecs_service
from dam.services import file_operations, hashing_service, tag_service, transcode_service

# Assuming test_environment fixture can be imported or replicated if needed.
# For now, let's use the one from test_cli by importing it.
# This might require adjustments if there are import cycles or if it's preferred to keep it separate.
from .test_cli import (
    _create_dummy_file,
    test_environment,  # noqa: F401
)

# Pytest marker for async tests
pytestmark = pytest.mark.asyncio


async def _add_dummy_asset(
    world: World, tmp_path: Path, filename: str = "source_asset.txt", content: str = "dummy source content"
) -> Entity:
    """Helper to add a dummy asset and return its entity."""
    dummy_file = _create_dummy_file(tmp_path / filename, content)

    original_filename, size_bytes = file_operations.get_file_properties(dummy_file)

    ingestion_event = AssetFileIngestionRequested(
        filepath_on_disk=dummy_file,
        original_filename=original_filename,
        size_bytes=size_bytes,
        world_name=world.name,
    )

    await world.dispatch_event(ingestion_event)
    await world.execute_stage(SystemStage.METADATA_EXTRACTION)

    # Find the entity by hash
    async with world.db_session_maker() as session:
        content_hash_val = hashing_service.calculate_sha256(dummy_file)
        # content_hash_bytes = bytes.fromhex(content_hash_val) # SHA256 in DB is string
        content_hash_bytes = bytes.fromhex(content_hash_val)

        stmt_hash = select(ContentHashSHA256Component).where(
            ContentHashSHA256Component.hash_value == content_hash_bytes  # Query with bytes
        )
        hash_comp_result = await session.execute(stmt_hash)
        hash_component = hash_comp_result.scalar_one_or_none()
        if not hash_component:
            raise LookupError(f"Failed to find hash component for {filename} after ingestion.")

        entity = await session.get(Entity, hash_component.entity_id)
        if not entity:
            raise LookupError(f"Failed to find entity for {filename} after ingestion.")
        return entity


# Test for transcode_service.create_transcode_profile
async def test_service_create_transcode_profile(test_environment):
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]

    current_test_settings = AppSettings()
    create_and_register_all_worlds_from_settings(app_settings=current_test_settings)
    target_world = get_world(default_world_name)
    assert target_world is not None

    profile_name = "test_service_profile"
    tool_name = "ffmpeg"
    parameters = "-crf 23"
    output_format = "mp4"
    description = "Test service profile description"

    profile_entity = await transcode_service.create_transcode_profile(
        world=target_world,
        profile_name=profile_name,
        tool_name=tool_name,
        parameters=parameters,
        output_format=output_format,
        description=description,
    )

    assert profile_entity is not None
    assert profile_entity.id is not None

    async with target_world.db_session_maker() as session:
        # Verify TranscodeProfileComponent
        tpc = await session.get(TranscodeProfileComponent, profile_entity.id)
        assert tpc is not None
        assert tpc.profile_name == profile_name
        assert tpc.tool_name == tool_name
        assert tpc.parameters == parameters
        assert tpc.output_format == output_format
        assert tpc.description == description
        assert tpc.concept_name == profile_name
        assert tpc.concept_description == description

        # Verify System:TranscodeProfile tag
        tag_concept = await tag_service.get_tag_concept_by_name(session, "System:TranscodeProfile")
        assert tag_concept is not None

        link_stmt = select(EntityTagLinkComponent).where(
            EntityTagLinkComponent.entity_id == profile_entity.id,
            EntityTagLinkComponent.tag_concept_entity_id == tag_concept.id,
        )
        link_result = await session.execute(link_stmt)
        link = link_result.scalar_one_or_none()
        assert link is not None, "System:TranscodeProfile tag was not applied."


# Test for CLI transcode profile-create
async def test_cli_transcode_profile_create(test_environment, monkeypatch):  # Removed click_runner, added monkeypatch
    default_world_name = test_environment["default_world_name"]

    profile_name = "test_cli_profile"
    tool_name = "cjxl"
    parameters = "-q 90"
    output_format = "jxl"
    description = "Test CLI profile description"

    # The CLI needs worlds to be initialized. test_environment does this.
    # We will call the command function directly due to async issues with CliRunner.

    captured_output = []

    def mock_secho(message="", **kwargs):  # Adjusted to match typer.secho signature
        captured_output.append(str(message))

    monkeypatch.setattr(typer, "secho", mock_secho)
    # If dam.cli directly imports and uses secho, patch it there too.
    # Assuming it uses typer.secho which is patched above.

    # Set global state that the command relies on (if main_callback doesn't run in this direct call context)
    # create_and_register_all_worlds_from_settings() is called by test_environment setup via main_callback,
    # but since we are calling the command function directly, we need to ensure worlds are registered.
    # app_config.settings should be patched by the test_environment fixture.
    create_and_register_all_worlds_from_settings(app_settings=dam_cli.app_config.settings)

    dam_cli.global_state.world_name = default_world_name
    target_world = get_world(default_world_name)
    assert target_world is not None

    mock_ctx = MagicMock(spec=typer.Context)
    # mock_ctx.obj = global_state # global_state is not defined here, but dam_cli.global_state is what matters
    mock_ctx.obj = dam_cli.global_state

    try:
        await dam_cli.cli_transcode_profile_create(
            ctx=mock_ctx,
            profile_name=profile_name,
            tool_name=tool_name,
            parameters=parameters,
            output_format=output_format,
            description=description,
        )
    except typer.Exit as e:
        # If the command calls typer.Exit, we need to check its code
        assert e.exit_code == 0, f"CLI command exited with code {e.exit_code}"

    output_str = "\n".join(captured_output)
    # print(f"Captured output for cli_transcode_profile_create: {output_str}") # For debugging

    assert f"Transcode profile '{profile_name}'" in output_str, f"Expected profile name in output. Got: {output_str}"
    assert "created successfully" in output_str, f"Expected success message in output. Got: {output_str}"

    # Verify in DB
    # target_world is already fetched
    assert target_world is not None
    async with target_world.db_session_maker() as session:
        stmt = select(TranscodeProfileComponent).where(TranscodeProfileComponent.profile_name == profile_name)
        db_profile_comp = (await session.execute(stmt)).scalar_one_or_none()

        assert db_profile_comp is not None
        assert db_profile_comp.tool_name == tool_name
        assert db_profile_comp.parameters == parameters
        assert db_profile_comp.output_format == output_format
        assert db_profile_comp.description == description
        assert db_profile_comp.concept_name == profile_name
        assert db_profile_comp.concept_description == description

        profile_entity_id = db_profile_comp.entity_id

        tag_concept = await tag_service.get_tag_concept_by_name(session, "System:TranscodeProfile")
        assert tag_concept is not None

        link_stmt = select(EntityTagLinkComponent).where(
            EntityTagLinkComponent.entity_id == profile_entity_id,
            EntityTagLinkComponent.tag_concept_entity_id == tag_concept.id,
        )
        link_result = await session.execute(link_stmt)
        link = link_result.scalar_one_or_none()
        assert link is not None, "System:TranscodeProfile tag was not applied via CLI."


# Test for transcode_service.apply_transcode_profile
async def test_service_apply_transcode_profile(test_environment, monkeypatch):
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]

    current_test_settings = AppSettings()
    create_and_register_all_worlds_from_settings(app_settings=current_test_settings)
    target_world = get_world(default_world_name)
    assert target_world is not None

    # 1. Add a source asset
    source_asset_entity = await _add_dummy_asset(target_world, tmp_path, "source_for_apply.txt", "transcode me")

    # 2. Create a transcode profile
    profile_name = "apply_test_profile_service"
    profile_entity = await transcode_service.create_transcode_profile(
        world=target_world,
        profile_name=profile_name,
        tool_name="mocktool",
        parameters="-mockparams {input} {output}",
        output_format="mock",
        description="Profile for service apply test",
    )

    # 3. Mock `dam.utils.media_utils.transcode_media`
    mock_transcoded_content = b"transcoded content"

    # This path will be created by the mock
    # It needs to be in a place the service expects, like settings.TRANSCODING_TEMP_DIR
    # Or ensure the mock creates it where it's told by `output_path` argument to `transcode_media`

    # The service determines temp_output_filepath like:
    # temp_output_filename = f"{Path(source_filepath).stem}_{profile_component.profile_name...}.{profile_component.output_format}"
    # temp_output_filepath = final_output_dir_base / temp_output_filename
    # We need to ensure our mock creates a file that the service can then find and ingest.

    # Let the mock handle file creation at the path it's given by the service
    async def mock_transcode_media_impl(input_path: Path, output_path: Path, tool_name: str, tool_params: str) -> Path:
        output_path.write_bytes(mock_transcoded_content)
        return output_path

    mock_transcode_media_async = AsyncMock(side_effect=mock_transcode_media_impl)
    monkeypatch.setattr(transcode_service, "transcode_media", mock_transcode_media_async)  # Patch where it's looked up

    # 4. Apply the profile
    transcoded_entity = await transcode_service.apply_transcode_profile(
        world=target_world, source_asset_entity_id=source_asset_entity.id, profile_entity_id=profile_entity.id
    )

    assert transcoded_entity is not None
    assert transcoded_entity.id != source_asset_entity.id

    # 5. Verify results
    async with target_world.db_session_maker() as session:
        # Verify TranscodedVariantComponent
        tvc_stmt = select(TranscodedVariantComponent).where(
            TranscodedVariantComponent.entity_id == transcoded_entity.id
        )
        tvc_res = await session.execute(tvc_stmt)
        tvc = tvc_res.scalar_one_or_none()

        assert tvc is not None
        assert tvc.original_asset_entity_id == source_asset_entity.id
        assert tvc.transcode_profile_entity_id == profile_entity.id
        assert tvc.transcoded_file_size_bytes == len(mock_transcoded_content)

        # Verify new asset's components (FPC, FLC, Hash)
        fpc = await dam_ecs_service.get_component(session, transcoded_entity.id, FilePropertiesComponent)
        assert fpc is not None
        source_fpc_for_name = await dam_ecs_service.get_component(
            session, source_asset_entity.id, FilePropertiesComponent
        )
        assert source_fpc_for_name is not None
        expected_new_filename_base = Path(source_fpc_for_name.original_filename).stem
        expected_new_filename = f"{expected_new_filename_base}_{profile_name.replace(' ', '_')}.mock"
        assert fpc.original_filename == expected_new_filename
        assert fpc.file_size_bytes == len(mock_transcoded_content)

        flc = await dam_ecs_service.get_component(session, transcoded_entity.id, FileLocationComponent)
        assert flc is not None
        assert "dam://local_cas/" in flc.url

        # Check that the mock was called
        mock_transcode_media_async.assert_called_once()
        # We could also check call args if needed, but it's complex due to temp paths.


# Test for CLI transcode apply
async def test_cli_transcode_apply(test_environment, monkeypatch):  # Removed click_runner
    tmp_path = test_environment["tmp_path"]
    default_world_name = test_environment["default_world_name"]

    current_test_settings = AppSettings()
    # Ensure worlds are registered using the settings patched by test_environment
    create_and_register_all_worlds_from_settings(
        app_settings=dam_cli.app_config.settings
    )  # Use settings from dam_cli's app_config

    dam_cli.global_state.world_name = default_world_name
    target_world = get_world(default_world_name)
    assert target_world is not None

    # 1. Add a source asset
    source_asset_entity = await _add_dummy_asset(target_world, tmp_path, "source_for_cli_apply.txt", "cli transcode me")

    # 2. Create a transcode profile (can use CLI or service for setup)
    profile_name_cli = "apply_test_profile_cli"
    # Using service to create profile to simplify setup for this test
    profile_entity_cli = await transcode_service.create_transcode_profile(
        world=target_world,
        profile_name=profile_name_cli,
        tool_name="mocktool_cli",
        parameters="-mockparams_cli {input} {output}",  # Added placeholders
        output_format="cli_mock",
    )

    # 3. Mock `dam.utils.media_utils.transcode_media`
    mock_transcoded_cli_content = b"cli transcoded content"

    async def mock_transcode_media_cli_impl(
        input_path: Path, output_path: Path, tool_name: str, tool_params: str
    ) -> Path:
        output_path.write_bytes(mock_transcoded_cli_content)
        return output_path

    mock_transcode_media_cli_async = AsyncMock(side_effect=mock_transcode_media_cli_impl)
    # Patch where it's looked up by the service (transcode_service calls dam.utils.media_utils.transcode_media)
    # The service transcode_service itself imports transcode_media from dam.utils.media_utils
    # So, we need to patch it there if transcode_service.py is the caller.
    # However, the apply_transcode_profile in transcode_service calls the transcode_media that *it* imported.
    # So, monkeypatch.setattr(transcode_service, "transcode_media", ...) was correct for the service test.
    # For the CLI test, if the CLI directly calls a utility or another service that calls transcode_media,
    # we need to patch it at the point of call or where it's imported by the direct caller from CLI.
    # The CLI command dam_cli.cli_transcode_apply calls transcode_service.apply_transcode_profile.
    # So, patching transcode_service.transcode_media is still the right place.
    monkeypatch.setattr(transcode_service, "transcode_media", mock_transcode_media_cli_async)

    captured_output_cli = []

    def mock_secho_cli(message="", **kwargs):
        captured_output_cli.append(str(message))

    monkeypatch.setattr(typer, "secho", mock_secho_cli)

    mock_ctx_cli = MagicMock(spec=typer.Context)
    mock_ctx_cli.obj = dam_cli.global_state

    # 4. Apply the profile via direct async call
    try:
        await dam_cli.cli_transcode_apply(
            ctx=mock_ctx_cli,
            asset_identifier=str(source_asset_entity.id),
            profile_identifier=str(profile_entity_cli.id),
            output_path_str=None,
        )
    except typer.Exit as e:
        assert e.exit_code == 0, f"CLI command exited with code {e.exit_code}"

    output_str_cli = "\n".join(captured_output_cli)
    assert "Transcoding successful." in output_str_cli, f"Actual output: {output_str_cli}"

    # Extract new entity ID from output if possible, or query
    # For now, let's query by finding the TVC

    async with target_world.db_session_maker() as session:
        # Find the TranscodedVariantComponent created for this operation
        tvc_stmt = (
            select(TranscodedVariantComponent)
            .where(
                TranscodedVariantComponent.original_asset_entity_id == source_asset_entity.id,
                TranscodedVariantComponent.transcode_profile_entity_id == profile_entity_cli.id,
            )
            .order_by(TranscodedVariantComponent.id.desc())
        )  # Get the latest one if multiple

        tvc_res = await session.execute(tvc_stmt)
        tvc = tvc_res.scalars().first()  # Use scalars().first()

        assert tvc is not None, "TranscodedVariantComponent not found after CLI apply."
        assert tvc.transcoded_file_size_bytes == len(mock_transcoded_cli_content)

        transcoded_entity_id_from_cli = tvc.entity_id
        assert transcoded_entity_id_from_cli is not None

        # Verify new asset's components
        fpc = await dam_ecs_service.get_component(session, transcoded_entity_id_from_cli, FilePropertiesComponent)
        assert fpc is not None
        source_fpc_for_name_cli = await dam_ecs_service.get_component(
            session, source_asset_entity.id, FilePropertiesComponent
        )
        assert source_fpc_for_name_cli is not None
        expected_new_filename_base_cli = Path(source_fpc_for_name_cli.original_filename).stem
        expected_new_filename_cli = f"{expected_new_filename_base_cli}_{profile_name_cli.replace(' ', '_')}.cli_mock"
        assert fpc.original_filename == expected_new_filename_cli
        assert fpc.file_size_bytes == len(mock_transcoded_cli_content)

        flc = await dam_ecs_service.get_component(session, transcoded_entity_id_from_cli, FileLocationComponent)
        assert flc is not None
        assert "dam://local_cas/" in flc.url

    mock_transcode_media_cli_async.assert_called_once()


# TODO: Add test for applying profile by name for asset and profile identifiers in CLI.
# TODO: Add test for error conditions, e.g., source asset not found, profile not found, transcoding failure.
