import traceback
import uuid
from dataclasses import dataclass
from typing import BinaryIO, List, Optional

import typer
from dam.core.commands import BaseCommand
from dam.core.world import get_world
from dam.functions import ecs_functions as dam_ecs_functions
from dam.models.core.entity import Entity
from typing_extensions import Annotated


@dataclass
class IngestAssetStreamCommand(BaseCommand):
    """A command to ingest a new asset from an in-memory stream."""

    entity: Entity
    file_content: BinaryIO
    original_filename: str
    world_name: str


@dataclass
class IngestAssetsCommand(BaseCommand[List[int]]):
    """A command to ingest new assets from a list of file paths."""

    file_paths: List[str]
    passwords: Optional[List[str]] = None


@dataclass
class GetAssetStreamCommand(BaseCommand):
    """A command to get a readable stream for an asset."""
    entity_id: int


@dataclass
class AutoTagEntityCommand(BaseCommand):
    """A command to trigger auto-tagging for an entity."""

    entity: Entity


async def cli_show_entity(
    ctx: typer.Context,
    entity_id: Annotated[
        int,
        typer.Argument(
            ...,
            help="The ID of the entity to show.",
        ),
    ],
):
    """
    Shows all components of a given entity.
    """
    if not ctx.obj.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    target_world = get_world(ctx.obj.world_name)
    if not target_world:
        typer.secho(f"Error: World '{ctx.obj.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    async with target_world.db_session_maker() as session:
        try:
            components = await dam_ecs_functions.get_all_components_for_entity(session, entity_id)
            if not components:
                typer.secho(f"No components found for entity {entity_id}", fg=typer.colors.YELLOW)
                return

            typer.secho(f"Components for entity {entity_id}:", fg=typer.colors.GREEN)
            for component in components:
                typer.echo(f"  - {component.__class__.__name__}:")
                for key, value in component.__dict__.items():
                    if not key.startswith("_"):
                        typer.echo(f"    - {key}: {value}")

        except Exception as e:
            typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            raise typer.Exit(code=1)
