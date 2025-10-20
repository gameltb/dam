"""CLI commands for dynamic world management."""

from dataclasses import dataclass
from typing import Annotated

import typer
from dam.commands.core import BaseCommand
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.world_manager import create_world_from_components
from dam.models.config import ConfigComponent
from dam.system_events.base import BaseSystemEvent

from dam_app.state import global_state

app = typer.Typer(
    name="world",
    help="Commands for dynamic world management.",
    add_completion=False,
    no_args_is_help=True,
)


@dataclass
class InstantiateWorldFromEntityCommand(BaseCommand[str, BaseSystemEvent]):
    """A command to instantiate a new world based on the components of an entity."""

    entity_id: int


@system(on_command=InstantiateWorldFromEntityCommand)
async def instantiate_world_from_entity_handler(
    cmd: InstantiateWorldFromEntityCommand,
    transaction: WorldTransaction,
) -> str:
    """Instantiate a new world from the components of an entity."""
    entity = await transaction.get_entity(cmd.entity_id)
    if not entity:
        raise ValueError(f"Entity with ID {cmd.entity_id} not found.")

    # Find all ConfigComponent subclasses attached to this entity
    config_components = await transaction.get_components(entity.id, ConfigComponent)

    if not config_components:
        raise ValueError(f"No ConfigComponents found on entity {cmd.entity_id}.")

    # Use the entity ID as the world name for simplicity
    world_name = f"world_from_entity_{cmd.entity_id}"

    # Create the world using the factory function
    new_world = create_world_from_components(world_name, config_components)

    return new_world.name


@app.command("from-entity")
async def from_entity_command(
    entity_id: Annotated[int, typer.Argument(help="The ID of the entity holding the world configuration.")],
    world: Annotated[str, typer.Option("--world", "-w", help="The name of the world to run this command in.")],
):
    """Instantiate a new DAM world based on the configuration components attached to an entity."""
    from dam.core.world import World  # noqa: PLC0415

    # We need an existing world to run the command in
    global_state.world_name = world
    active_world: World | None = global_state.get_current_world()
    if not active_world:
        typer.secho(f"Error: Could not instantiate the active world '{world}'.", fg=typer.colors.RED)
        raise typer.Exit(1)

    import typing  # noqa: PLC0415
    from typing import Any  # noqa: PLC0415

    # Dispatch the command and wait for the result
    command = InstantiateWorldFromEntityCommand(entity_id=entity_id)
    result = ""
    async for res in typing.cast(Any, active_world).execute_command(command):
        result = res

    typer.secho(f"Successfully instantiated new world: '{result}'", fg=typer.colors.GREEN)
