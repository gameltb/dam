from pathlib import Path

import pytest
from dam.core.world import World
from dam.functions import ecs_functions

from dam_fs.commands import IngestFileCommand
from dam_fs.models import FilePropertiesComponent


@pytest.mark.serial
@pytest.mark.asyncio
async def test_handle_ingest_file_command(test_world_alpha: World, temp_asset_file: Path):
    """Test the handle_ingest_file_command system."""

    from dam.core.transaction import EcsTransaction, active_transaction

    async with test_world_alpha.db_session_maker() as session:
        transaction = EcsTransaction(session)
        token = active_transaction.set(transaction)
        try:
            command = IngestFileCommand(
                filepath_on_disk=temp_asset_file,
                original_filename=temp_asset_file.name,
                size_bytes=temp_asset_file.stat().st_size,
                world_name=test_world_alpha.name,
            )
            await test_world_alpha.dispatch_command(command)
            await session.commit()
        finally:
            active_transaction.reset(token)

    # Verify that the asset was added
    async with test_world_alpha.db_session_maker() as session:
        entities = await ecs_functions.find_entities_with_components(session, [FilePropertiesComponent])
        assert len(entities) == 1
        entity = entities[0]
        fp_component = await ecs_functions.get_component(session, entity.id, FilePropertiesComponent)
        assert fp_component is not None
        assert fp_component.original_filename == temp_asset_file.name
