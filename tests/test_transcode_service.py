import asyncio
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from dam.core.world import World
from dam.core.database import DatabaseManager
from dam.core.config import WorldConfig, Settings
from dam.models.core.entity import Entity
from dam.models.core.base_component import Base
from dam.models.conceptual.transcode_profile_component import TranscodeProfileComponent
from dam.models.conceptual.transcoded_variant_component import TranscodedVariantComponent
from dam.models.core.file_location_component import FileLocationComponent
from dam.models.core.file_properties_component import FilePropertiesComponent
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component

from dam.services import transcode_service, ecs_service, file_operations, tag_service
from dam.utils.media_utils import TranscodeError, transcode_media as actual_transcode_media
from dam.core.world_setup import initialize_world_resources, register_core_systems
from dam.core.events import AssetFileIngestionRequested

# Database setup for tests
DATABASE_URL = "sqlite+aiosqlite:///./test_transcode_service.db"
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False) # type: ignore

# Global settings for test
test_settings = Settings(
    DAM_WORLDS_CONFIG=f'{{"test_world": {{"DATABASE_URL": "{DATABASE_URL}", "ASSET_STORAGE_PATH": "./test_dam_storage_transcode"}}}}',
    DEFAULT_WORLD_NAME="test_world",
    TESTING_MODE=True,
    TRANSCODING_TEMP_DIR="temp/test_transcodes_transcode_svc"
)

@pytest_asyncio.fixture(scope="module")
async def setup_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    Path("./test_transcode_service.db").unlink(missing_ok=True)

@pytest_asyncio.fixture
async def db_session(setup_database):
    async with AsyncSessionLocal() as session:
        yield session

@pytest_asyncio.fixture
async def test_world(tmp_path_factory):
    # Create unique storage and temp paths for each test world instance if needed,
    # or manage cleanup carefully. For this fixture, let's use one set by test_settings.
    storage_path = Path(test_settings.get_world_config("test_world").ASSET_STORAGE_PATH)
    storage_path.mkdir(parents=True, exist_ok=True)

    transcode_temp_path = Path(test_settings.TRANSCODING_TEMP_DIR)
    transcode_temp_path.mkdir(parents=True, exist_ok=True)

    world_config = test_settings.get_world_config("test_world")

    # Create a new world instance for each test to ensure isolation of resources and scheduler state
    world = World(name="test_world", config=world_config, component_registry=None, settings_override=test_settings)

    initialize_world_resources(world) # Populates db_manager, etc.
    register_core_systems(world) # Registers system event handlers

    # Ensure the database is clean for this world's db_manager instance for each test
    # This is important if tests run concurrently or if previous tests leave data
    async with world.db_manager.engine.begin() as conn: # type: ignore
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield world

    # Cleanup (optional, if test_settings paths are specific enough per test run or module)
    # shutil.rmtree(storage_path, ignore_errors=True)
    # shutil.rmtree(transcode_temp_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_create_transcode_profile(test_world: World, db_session: AsyncSession):
    profile_name = "test_h264_medium"
    tool = "ffmpeg"
    params = "-i {input} -c:v libx264 -preset medium -crf 23 {output}"
    output_format = "mp4"
    description = "H.264 Medium Quality"

    profile_entity = await transcode_service.create_transcode_profile(
        world=test_world,
        profile_name=profile_name,
        tool_name=tool,
        parameters=params,
        output_format=output_format,
        description=description,
        # session=db_session # Service should manage its own session
    )

    assert profile_entity is not None
    assert profile_entity.id is not None

    async with test_world.db_session_maker() as session: # Use world's session for verification
        # Verify Entity
        retrieved_entity = await session.get(Entity, profile_entity.id)
        assert retrieved_entity is not None

        # Verify TranscodeProfileComponent
        stmt = select(TranscodeProfileComponent).where(TranscodeProfileComponent.entity_id == profile_entity.id)
        profile_comp = (await session.execute(stmt)).scalars().first()

        assert profile_comp is not None
        assert profile_comp.profile_name == profile_name
        assert profile_comp.tool_name == tool
        assert profile_comp.parameters == params
        assert profile_comp.output_format == output_format
        assert profile_comp.description == description
        assert profile_comp.concept_name == profile_name # from BaseConceptualInfo

        # Verify system tag was applied (optional, but good to check)
        tags = await tag_service.get_tags_for_entity(test_world, entity_id=profile_entity.id, session=session)
        assert any(tag.name == "System:TranscodeProfile" for tag, _ in tags)


@pytest.mark.asyncio
async def test_create_transcode_profile_already_exists(test_world: World, db_session: AsyncSession):
    profile_name = "test_h264_fast_unique"
    await transcode_service.create_transcode_profile(
        test_world, profile_name, "ffmpeg", "-c:v libx264 {output}", "mp4"
    )
    with pytest.raises(transcode_service.TranscodeServiceError, match=f"Transcode profile '{profile_name}' already exists"):
        await transcode_service.create_transcode_profile(
            test_world, profile_name, "ffmpeg", "-c:v libx264 {output}", "mp4"
        )

@pytest.mark.asyncio
async def test_get_transcode_profile_by_name_or_id(test_world: World, db_session: AsyncSession):
    profile_name = "test_av1_cq20"
    tool = "avifenc" # Placeholder, actual tool might be different for AV1 video
    params = "-c:v libaom-av1 -crf 20 {output}" # Example
    output_format = "mp4" # or mkv, webm for AV1 video

    created_profile_entity = await transcode_service.create_transcode_profile(
        test_world, profile_name, tool, params, output_format
    )

    # Get by name
    entity_by_name, comp_by_name = await transcode_service.get_transcode_profile_by_name_or_id(
        test_world, profile_name # session=db_session
    )
    assert entity_by_name.id == created_profile_entity.id
    assert comp_by_name.profile_name == profile_name

    # Get by ID
    entity_by_id, comp_by_id = await transcode_service.get_transcode_profile_by_name_or_id(
        test_world, created_profile_entity.id #, session=db_session
    )
    assert entity_by_id.id == created_profile_entity.id
    assert comp_by_id.profile_name == profile_name

    # Test not found
    with pytest.raises(transcode_service.TranscodeServiceError, match="Transcode profile 'non_existent_profile' not found"):
        await transcode_service.get_transcode_profile_by_name_or_id(test_world, "non_existent_profile") #, session=db_session)

    with pytest.raises(transcode_service.TranscodeServiceError, match="Transcode profile '99999' not found"): # Assuming 99999 is not a valid ID
        await transcode_service.get_transcode_profile_by_name_or_id(test_world, 99999) #, session=db_session)


@pytest_asyncio.fixture
async def sample_asset_entity(test_world: World, tmp_path: Path):
    # Create a dummy file to represent an asset
    dummy_file_content = b"This is a test file for transcoding."
    source_file = tmp_path / "source_video.mp4" # Keep extension for mime type if possible
    source_file.write_bytes(dummy_file_content)

    # Ingest this dummy file to create an asset entity
    file_props = file_operations.get_file_properties(source_file)

    # Mock the event dispatch and subsequent processing for simplicity in this unit test fixture
    # In a fuller integration test, you'd let the events run.
    async with test_world.db_session_maker() as session:
        entity = Entity()
        session.add(entity)
        await session.flush()

        fpc = FilePropertiesComponent(
            entity_id=entity.id,
            original_filename=source_file.name,
            file_size_bytes=file_props.size_bytes, # type: ignore
            mime_type=file_props.mime_type, # type: ignore
        )
        session.add(fpc)

        # For apply_transcode_profile, we need a FileLocationComponent
        # Simulate that it's stored in DAM storage (even if just a reference to temp file for test)
        # The actual storage path for the "source" asset:
        dam_storage_path_for_source = Path(test_world.config.ASSET_STORAGE_PATH) / "test_source_content_id"
        dam_storage_path_for_source.parent.mkdir(parents=True, exist_ok=True)
        dam_storage_path_for_source.write_bytes(source_file.read_bytes())


        flc = FileLocationComponent(
            entity_id=entity.id,
            storage_type="local_dam", # simulate it's in DAM storage
            physical_path_or_key=str(dam_storage_path_for_source), # Needs to be readable by transcode_media
            contextual_filename=source_file.name,
            content_identifier="test_source_content_id" # Made up for test
        )
        session.add(flc)

        sha256_hash = file_operations.calculate_sha256(source_file)
        hash_comp = ContentHashSHA256Component(entity_id=entity.id, hash_value=sha256_hash)
        session.add(hash_comp)

        await session.commit()
        await session.refresh(entity)
        return entity.id, source_file # Return ID and original path for reference


@pytest.mark.asyncio
@patch("dam.services.transcode_service.transcode_media", autospec=True) # Mock the actual transcoding utility
@patch("dam.core.world.World.dispatch_event", new_callable=AsyncMock) # Mock event dispatch
async def test_apply_transcode_profile(
    mock_dispatch_event: AsyncMock,
    mock_transcode_media: MagicMock,
    test_world: World,
    sample_asset_entity: tuple[int, Path],
    tmp_path: Path # For the output of the mocked transcode_media
):
    source_asset_entity_id, original_source_filepath = sample_asset_entity

    # 1. Create a transcode profile
    profile_name = "test_apply_profile"
    profile_entity = await transcode_service.create_transcode_profile(
        test_world, profile_name, "ffmpeg", "-i {input} -f webm {output}", "webm"
    )

    # 2. Configure mock for transcode_media
    # It should create a dummy output file where it's told to.
    # The service determines the temp output path. We need to know what it will be or make it predictable.
    # Let's assume the service creates a unique name in TRANSCODING_TEMP_DIR.
    # The mock needs to return the path to this "transcoded" file.

    # Predictable output from the mocked transcode_media
    # This path needs to be inside test_world.settings.TRANSCODING_TEMP_DIR
    # The actual name is generated by the service, e.g. source_stem_profilename_uuid.format
    # We can make mock_transcode_media create this file.

    mocked_transcoded_content = b"fake transcoded webm content"

    # This function will be called by the mock
    def side_effect_transcode_media(input_path, output_path, tool_name, tool_params):
        # output_path is determined by the service. The mock should write to it.
        output_path.write_bytes(mocked_transcoded_content)
        return output_path # Return the path where it wrote the file

    mock_transcode_media.side_effect = side_effect_transcode_media

    # 3. Configure mock for event dispatch (AssetFileIngestionRequested)
    # The event handler for AssetFileIngestionRequested would normally create the new entity, FPC, FLC, Hashes.
    # We need to simulate this behavior so apply_transcode_profile can find the new entity.

    # This is tricky because apply_transcode_profile dispatches an event and then queries based on hash.
    # The event handler (handle_asset_file_ingestion_request) runs in its own transaction context usually.
    # For this test, we'll have the mock_dispatch_event directly create the necessary components
    # in the DB using the test_world's session, as if the event handler ran successfully and immediately.

    async def simulate_ingestion_event_handling(event: AssetFileIngestionRequested):
        if isinstance(event, AssetFileIngestionRequested):
            # This is the file created by the mocked transcode_media
            transcoded_filepath_by_mock = event.filepath_on_disk

            async with test_world.db_session_maker() as session:
                new_entity = Entity()
                session.add(new_entity)
                await session.flush()

                fpc = FilePropertiesComponent(
                    entity_id=new_entity.id,
                    original_filename=event.original_filename,
                    file_size_bytes=event.size_bytes,
                    mime_type=event.mime_type
                )
                session.add(fpc)

                # Simulate storage in DAM (content-addressable)
                # For the test, the "transcoded_filepath_by_mock" is temporary.
                # The "ingestion" should place it into the DAM's actual storage.
                dam_storage_base = Path(test_world.config.ASSET_STORAGE_PATH)
                content_id = file_operations.calculate_sha256(transcoded_filepath_by_mock) # Hash of the mocked content

                # This path is where the DAM would store it.
                final_dam_path = dam_storage_base / content_id[:2] / content_id
                final_dam_path.parent.mkdir(parents=True, exist_ok=True)
                # "Copy" the mocked content to this final path
                final_dam_path.write_bytes(transcoded_filepath_by_mock.read_bytes())


                flc = FileLocationComponent(
                    entity_id=new_entity.id,
                    storage_type="local_dam",
                    physical_path_or_key=str(final_dam_path),
                    contextual_filename=event.original_filename, # Or a DAM derived one
                    content_identifier=content_id
                )
                session.add(flc)

                hash_comp = ContentHashSHA256Component(entity_id=new_entity.id, hash_value=content_id)
                session.add(hash_comp)

                await session.commit()
                # No need to return anything, apply_transcode_profile will query by hash.

    mock_dispatch_event.side_effect = simulate_ingestion_event_handling

    # 4. Call the service function
    transcoded_entity = await transcode_service.apply_transcode_profile(
        world=test_world,
        source_asset_entity_id=source_asset_entity_id,
        profile_entity_id=profile_entity.id,
        # output_parent_dir=tmp_path # Not needed if service uses its temp dir
    )

    # 5. Assertions
    assert transcoded_entity is not None
    assert transcoded_entity.id is not None

    # Verify mock_transcode_media was called correctly
    # The output path for mock_transcode_media is generated inside the service.
    # We can check that its parent is the TRANSCODING_TEMP_DIR
    assert mock_transcode_media.call_count == 1
    args, _ = mock_transcode_media.call_args
    assert args[0] == await transcode_service._get_source_asset_filepath(test_world, source_asset_entity_id, await test_world.db_session_maker().__anext__()) # type: ignore
    assert args[1].parent == Path(test_settings.TRANSCODING_TEMP_DIR) # Check output path parent
    assert args[2] == "ffmpeg" # tool_name
    assert args[3] == "-i {input} -f webm {output}" # tool_params

    # Verify mock_dispatch_event was called with AssetFileIngestionRequested
    assert mock_dispatch_event.call_count == 1
    dispatched_event = mock_dispatch_event.call_args[0][0]
    assert isinstance(dispatched_event, AssetFileIngestionRequested)
    # dispatched_event.filepath_on_disk will be the path mock_transcode_media wrote to.
    # Check its content hash matches the hash of mocked_transcoded_content
    expected_hash_of_mocked_output = file_operations.calculate_sha256_from_bytes(mocked_transcoded_content)

    async with test_world.db_session_maker() as session:
        # Verify TranscodedVariantComponent
        stmt_variant = select(TranscodedVariantComponent).where(TranscodedVariantComponent.entity_id == transcoded_entity.id)
        variant_comp = (await session.execute(stmt_variant)).scalars().first()
        assert variant_comp is not None
        assert variant_comp.original_asset_entity_id == source_asset_entity_id
        assert variant_comp.transcode_profile_entity_id == profile_entity.id
        assert variant_comp.transcoded_file_size_bytes == len(mocked_transcoded_content)

        # Verify the new entity has the correct hash (based on mocked content)
        stmt_hash = select(ContentHashSHA256Component).where(ContentHashSHA256Component.entity_id == transcoded_entity.id)
        retrieved_hash_comp = (await session.execute(stmt_hash)).scalars().first()
        assert retrieved_hash_comp is not None
        assert retrieved_hash_comp.hash_value == expected_hash_of_mocked_output

        # Verify FilePropertiesComponent for the new entity
        stmt_fpc = select(FilePropertiesComponent).where(FilePropertiesComponent.entity_id == transcoded_entity.id)
        retrieved_fpc = (await session.execute(stmt_fpc)).scalars().first()
        assert retrieved_fpc is not None
        assert retrieved_fpc.file_size_bytes == len(mocked_transcoded_content)
        # Mime type would depend on mocked file_operations.get_file_properties(mocked_output_file)
        # The original_filename is derived by the service.
        source_fpc_orig = await ecs_service.get_component_for_entity(session, source_asset_entity_id, FilePropertiesComponent) # type: ignore
        expected_orig_name_base = Path(source_fpc_orig.original_filename).stem # type: ignore
        expected_new_asset_name = f"{expected_orig_name_base}_{profile_name.replace(' ', '_')}.webm"
        assert retrieved_fpc.original_filename == expected_new_asset_name

        # Verify FileLocationComponent points to the "DAM stored" path (simulated by mock_dispatch_event)
        stmt_flc = select(FileLocationComponent).where(FileLocationComponent.entity_id == transcoded_entity.id)
        retrieved_flc = (await session.execute(stmt_flc)).scalars().first()
        assert retrieved_flc is not None
        assert retrieved_flc.content_identifier == expected_hash_of_mocked_output
        expected_final_dam_path = Path(test_world.config.ASSET_STORAGE_PATH) / expected_hash_of_mocked_output[:2] / expected_hash_of_mocked_output
        assert Path(retrieved_flc.physical_path_or_key) == expected_final_dam_path
        assert expected_final_dam_path.exists() # Check the mock ingestion actually "stored" it
        assert expected_final_dam_path.read_bytes() == mocked_transcoded_content


@pytest.mark.asyncio
async def test_apply_transcode_profile_transcoding_fails(
    test_world: World,
    sample_asset_entity: tuple[int, Path],
):
    source_asset_entity_id, _ = sample_asset_entity
    profile_entity = await transcode_service.create_transcode_profile(
        test_world, "failing_profile", "nonexistenttool", "{input} {output}", "fail"
    )

    with patch("dam.services.transcode_service.transcode_media", side_effect=TranscodeError("Tool not found")):
        with pytest.raises(transcode_service.TranscodeServiceError, match="Transcoding failed: Tool not found"):
            await transcode_service.apply_transcode_profile(
                world=test_world,
                source_asset_entity_id=source_asset_entity_id,
                profile_entity_id=profile_entity.id,
            )

@pytest.mark.asyncio
async def test_get_transcoded_variants_for_original(
    test_world: World,
    sample_asset_entity: tuple[int, Path],
    tmp_path: Path
):
    source_asset_id, _ = sample_asset_entity

    # Create two profiles
    profile1 = await transcode_service.create_transcode_profile(test_world, "tp_var1", "tool1", "{i} {o}", "fmt1")
    profile2 = await transcode_service.create_transcode_profile(test_world, "tp_var2", "tool2", "{i} {o}", "fmt2")

    # Mock actual transcoding and ingestion as in test_apply_transcode_profile
    with patch("dam.services.transcode_service.transcode_media", autospec=True) as mock_tm, \
         patch("dam.core.world.World.dispatch_event", new_callable=AsyncMock) as mock_de:

        # Common side effect for transcode_media mock
        def side_effect_tm(input_path, output_path, tool_name, tool_params):
            output_path.write_bytes(f"content for {tool_name}".encode())
            return output_path
        mock_tm.side_effect = side_effect_tm

        # Common side effect for dispatch_event mock (simulating ingestion)
        async def simulate_ingestion(event: AssetFileIngestionRequested):
            async with test_world.db_session_maker() as session:
                new_e = Entity()
                session.add(new_e)
                await session.flush()
                # Add minimal components for the test to pass
                fpc = FilePropertiesComponent(entity_id=new_e.id, original_filename=event.original_filename, file_size_bytes=event.size_bytes, mime_type=event.mime_type)
                session.add(fpc)
                hval = file_operations.calculate_sha256(event.filepath_on_disk)
                hash_c = ContentHashSHA256Component(entity_id=new_e.id, hash_value=hval)
                session.add(hash_c)
                # Simulate storage for FLC
                dam_storage_base = Path(test_world.config.ASSET_STORAGE_PATH)
                final_dam_path = dam_storage_base / hval[:2] / hval
                final_dam_path.parent.mkdir(parents=True, exist_ok=True)
                final_dam_path.write_bytes(event.filepath_on_disk.read_bytes())
                flc = FileLocationComponent(entity_id=new_e.id, storage_type="local_dam", physical_path_or_key=str(final_dam_path), content_identifier=hval)
                session.add(flc)
                await session.commit()
        mock_de.side_effect = simulate_ingestion

        # Apply profile 1
        transcoded1 = await transcode_service.apply_transcode_profile(test_world, source_asset_id, profile1.id)
        # Apply profile 2
        transcoded2 = await transcode_service.apply_transcode_profile(test_world, source_asset_id, profile2.id)

    variants = await transcode_service.get_transcoded_variants_for_original(test_world, source_asset_id)

    assert len(variants) == 2
    variant_entity_ids = {v[0].id for v in variants}
    assert transcoded1.id in variant_entity_ids
    assert transcoded2.id in variant_entity_ids

    for entity, var_comp, prof_comp in variants:
        assert var_comp.original_asset_entity_id == source_asset_id
        if entity.id == transcoded1.id:
            assert prof_comp.id == profile1.id
        elif entity.id == transcoded2.id:
            assert prof_comp.id == profile2.id
        else:
            pytest.fail("Unexpected transcoded entity found.")

@pytest.mark.asyncio
async def test_get_assets_using_profile(
    test_world: World,
    sample_asset_entity: tuple[int, Path], # Fixture provides one source asset
    tmp_path: Path
):
    source_asset_id1, _ = sample_asset_entity

    # Create another source asset for diversity
    dummy_content2 = b"another source file"
    source_file2 = tmp_path / "source2.txt"
    source_file2.write_bytes(dummy_content2)
    async with test_world.db_session_maker() as session:
        source_entity2 = Entity()
        session.add(source_entity2)
        await session.flush()
        source_asset_id2 = source_entity2.id
        fpc2 = FilePropertiesComponent(entity_id=source_asset_id2, original_filename="s2.txt", file_size_bytes=len(dummy_content2), mime_type="text/plain")
        session.add(fpc2)
        # Minimal FLC for _get_source_asset_filepath
        flc_path2 = Path(test_world.config.ASSET_STORAGE_PATH) / "source2_content_id"
        flc_path2.parent.mkdir(parents=True, exist_ok=True)
        flc_path2.write_bytes(source_file2.read_bytes())
        flc2 = FileLocationComponent(entity_id=source_asset_id2, storage_type="local_dam", physical_path_or_key=str(flc_path2), content_identifier="source2_content_id")
        session.add(flc2)
        await session.commit()


    profile_shared = await transcode_service.create_transcode_profile(test_world, "tp_shared", "tool_s", "{i} {o}", "sfmt")
    profile_other = await transcode_service.create_transcode_profile(test_world, "tp_other", "tool_o", "{i} {o}", "ofmt")

    with patch("dam.services.transcode_service.transcode_media", autospec=True) as mock_tm, \
         patch("dam.core.world.World.dispatch_event", new_callable=AsyncMock) as mock_de:

        # Common side effects for mocks
        def side_effect_tm(input_path, output_path, tool_name, tool_params):
            output_path.write_bytes(f"content by {tool_name}".encode())
            return output_path
        mock_tm.side_effect = side_effect_tm
        async def simulate_ingestion(event: AssetFileIngestionRequested): # Simplified
            async with test_world.db_session_maker() as session:
                new_e = Entity(); session.add(new_e); await session.flush()
                fpc = FilePropertiesComponent(entity_id=new_e.id, original_filename=event.original_filename, file_size_bytes=event.size_bytes, mime_type=event.mime_type); session.add(fpc)
                hval = file_operations.calculate_sha256(event.filepath_on_disk)
                hash_c = ContentHashSHA256Component(entity_id=new_e.id, hash_value=hval); session.add(hash_c)
                dam_storage_base = Path(test_world.config.ASSET_STORAGE_PATH)
                final_dam_path = dam_storage_base / hval[:2] / hval
                final_dam_path.parent.mkdir(parents=True, exist_ok=True)
                final_dam_path.write_bytes(event.filepath_on_disk.read_bytes())
                flc = FileLocationComponent(entity_id=new_e.id, storage_type="local_dam", physical_path_or_key=str(final_dam_path), content_identifier=hval); session.add(flc)
                await session.commit()
        mock_de.side_effect = simulate_ingestion

        # Asset 1 uses shared_profile
        transcoded_a1_p_shared = await transcode_service.apply_transcode_profile(test_world, source_asset_id1, profile_shared.id)
        # Asset 2 uses shared_profile
        transcoded_a2_p_shared = await transcode_service.apply_transcode_profile(test_world, source_asset_id2, profile_shared.id)
        # Asset 1 uses other_profile (should not be found when querying by shared_profile)
        _ = await transcode_service.apply_transcode_profile(test_world, source_asset_id1, profile_other.id)

    assets_with_shared_profile = await transcode_service.get_assets_using_profile(test_world, profile_shared.id)

    assert len(assets_with_shared_profile) == 2
    found_entity_ids = {e.id for e, _ in assets_with_shared_profile}
    assert transcoded_a1_p_shared.id in found_entity_ids
    assert transcoded_a2_p_shared.id in found_entity_ids

    for entity, var_comp in assets_with_shared_profile:
        assert var_comp.transcode_profile_entity_id == profile_shared.id
        # Check that these are indeed linked to the correct original assets
        if entity.id == transcoded_a1_p_shared.id:
            assert var_comp.original_asset_entity_id == source_asset_id1
        elif entity.id == transcoded_a2_p_shared.id:
            assert var_comp.original_asset_entity_id == source_asset_id2
        else:
            pytest.fail("Unexpected entity found using shared profile.")

    assets_with_other_profile = await transcode_service.get_assets_using_profile(test_world, profile_other.id)
    assert len(assets_with_other_profile) == 1
    assert assets_with_other_profile[0][0].id != transcoded_a1_p_shared.id # Ensure it's the one from other_profile
    assert assets_with_other_profile[0][0].id != transcoded_a2_p_shared.id
    assert assets_with_other_profile[0][1].original_asset_entity_id == source_asset_id1
    assert assets_with_other_profile[0][1].transcode_profile_entity_id == profile_other.id

# TODO: Add tests for _get_source_asset_filepath helper if its logic becomes more complex
# (e.g., resolving relative paths from different storage contexts). Current version is simple.
