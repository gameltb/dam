import pytest
from pathlib import Path

from dam.core.world import World
from dam_fs.commands import IngestFileCommand
from dam_media_audio.models.properties import AudioPropertiesComponent
from dam.functions import ecs_functions
from dam.core.stages import SystemStage

@pytest.mark.serial
@pytest.mark.asyncio
async def test_add_audio_components_system(test_world_alpha: World, sample_wav_file: Path):
    """Test the add_audio_components_system."""

    from dam.core.transaction import EcsTransaction, active_transaction

    async with test_world_alpha.db_session_maker() as session:
        transaction = EcsTransaction(session)
        token = active_transaction.set(transaction)
        try:
            command = IngestFileCommand(
                filepath_on_disk=sample_wav_file,
                original_filename=sample_wav_file.name,
                size_bytes=sample_wav_file.stat().st_size,
                world_name=test_world_alpha.name,
            )
            await test_world_alpha.dispatch_command(command)
            await session.commit()
        finally:
            active_transaction.reset(token)

    await test_world_alpha.execute_stage(SystemStage.METADATA_EXTRACTION)

    # Verify that the asset was added
    async with test_world_alpha.db_session_maker() as session:
        entities = await ecs_functions.find_entities_with_components(session, [AudioPropertiesComponent])
        assert len(entities) == 1
        entity = entities[0]
        audio_props = await ecs_functions.get_component(session, entity.id, AudioPropertiesComponent)
        assert audio_props is not None
        assert audio_props.duration_seconds == 1.0
        assert audio_props.codec_name == "pcm_s16le"
        assert audio_props.sample_rate_hz == 48000
        assert audio_props.channels == 1
