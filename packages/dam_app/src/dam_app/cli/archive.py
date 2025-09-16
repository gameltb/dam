from pathlib import Path
from typing import List

import typer
from dam_archive.commands import (
    ClearArchiveComponentsCommand,
    CreateMasterArchiveCommand,
    DiscoverAndBindCommand,
    UnbindSplitArchiveCommand,
)
from typing_extensions import Annotated

from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper(
    name="archive",
    help="Commands for managing archive assets.",
)


@app.command(name="clear-info")
async def clear_archive_info(
    entity_id: Annotated[int, typer.Argument(..., help="The ID of the archive entity to clear.")],
):
    """
    Removes archive-related components from an entity and its members.

    This is useful when you want to re-process an archive from scratch.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    typer.echo(f"Clearing archive info for entity: {entity_id}...")

    clear_cmd = ClearArchiveComponentsCommand(entity_id=entity_id)
    await target_world.dispatch_command(clear_cmd).get_all_results()

    typer.secho("Archive info clearing process complete.", fg=typer.colors.GREEN)


@app.command(name="discover-and-bind")
async def discover_and_bind(
    paths: Annotated[
        List[Path],
        typer.Argument(
            ...,
            help="Path to the asset file or directory to scan.",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
):
    """
    Scans paths for split archive parts, tags them, and binds complete sets.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    typer.echo(f"Discovering and binding split archives in paths: {paths}...")

    discover_cmd = DiscoverAndBindCommand(paths=[str(p) for p in paths])
    await target_world.dispatch_command(discover_cmd).get_all_results()

    typer.secho("Discovery and binding process complete.", fg=typer.colors.GREEN)


@app.command(name="create-master")
async def create_master(
    name: Annotated[str, typer.Option("--name", "-n", help="The name for the master archive entity.")],
    part_ids: Annotated[List[int], typer.Argument(..., help="An ordered list of entity IDs for the parts.")],
):
    """
    Manually creates a master entity for a split archive from a list of parts.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    typer.echo(f"Creating master archive '{name}' with parts: {part_ids}...")

    create_cmd = CreateMasterArchiveCommand(name=name, part_entity_ids=part_ids)
    await target_world.dispatch_command(create_cmd).get_all_results()

    typer.secho("Master archive created successfully.", fg=typer.colors.GREEN)


@app.command(name="unbind-master")
async def unbind_master(
    master_id: Annotated[int, typer.Argument(..., help="The entity ID of the master archive to unbind.")],
):
    """
    Unbinds a split archive, removing the master entity's manifest and part info.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    typer.echo(f"Unbinding master archive with ID: {master_id}...")

    unbind_cmd = UnbindSplitArchiveCommand(master_entity_id=master_id)
    await target_world.dispatch_command(unbind_cmd).get_all_results()

    typer.secho("Master archive unbound successfully.", fg=typer.colors.GREEN)
