from typing import Optional

import pytest
from typer.testing import CliRunner

from dam.core.world import World
from dam.models.properties import FilePropertiesComponent
from dam.services import ecs_service

runner = CliRunner()


@pytest.fixture(autouse=True)
async def current_test_world_for_inspect_cli(test_world_alpha: World) -> World:
    yield test_world_alpha


@pytest.mark.asyncio
async def test_cli_inspect_entity(current_test_world_for_inspect_cli: World, click_runner: CliRunner):
    world = current_test_world_for_inspect_cli
    world_name = world.name

    entity_id: Optional[int] = None
    async with world.db_session_maker() as session:
        entity = await ecs_service.create_entity(session)
        await ecs_service.add_component_to_entity(
            session,
            entity.id,
            FilePropertiesComponent(original_filename="inspect_test.txt", file_size_bytes=123),
        )
        await session.commit()
        entity_id = entity.id

    assert entity_id is not None

    from unittest.mock import patch

    from dam.cli import cli_inspect_entity, global_state

    global_state.world_name = world_name

    with patch("rich.print_json") as mock_print_json:
        await cli_inspect_entity(ctx=None, entity_id=entity_id)

        mock_print_json.assert_called_once()
        output_data = mock_print_json.call_args[1]["data"]

        assert "FilePropertiesComponent" in output_data
        assert len(output_data["FilePropertiesComponent"]) == 1

        fpc_data = output_data["FilePropertiesComponent"][0]
        assert fpc_data["original_filename"] == "inspect_test.txt"
        assert fpc_data["file_size_bytes"] == 123
