import pytest
from pathlib import Path

from dam.core.world import World

@pytest.mark.serial
@pytest.mark.asyncio
async def test_add_asset(test_world_alpha: World, temp_asset_file: Path):
    """Test the add_asset function."""
    from dam_app.cli import add_asset

    await add_asset(test_world_alpha, [temp_asset_file])

    # Verify that the asset was added
    from dam_fs.models import FilePropertiesComponent
    from dam.functions import ecs_functions

    async with test_world_alpha.db_session_maker() as session:
        entities = await ecs_functions.find_entities_with_components(session, [FilePropertiesComponent])
        assert len(entities) == 1
        entity = entities[0]
        fp_component = await ecs_functions.get_component(session, entity.id, FilePropertiesComponent)
        assert fp_component is not None
        assert fp_component.original_filename == temp_asset_file.name


@pytest.mark.serial
def test_cli_list_worlds(settings_override, capsys):
    """Test the list-worlds command."""
    from dam_app.cli import cli_list_worlds, create_and_register_all_worlds_from_settings

    # Ensure worlds are registered
    create_and_register_all_worlds_from_settings(settings_override)

    cli_list_worlds()

    captured = capsys.readouterr()
    assert "test_world_alpha" in captured.out
    assert "test_world_beta" in captured.out
