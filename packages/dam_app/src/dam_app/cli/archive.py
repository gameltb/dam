from typing import List

import typer
from dam_archive.commands import (
    CreateMasterArchiveCommand,
    DiscoverAndBindCommand,
    UnbindSplitArchiveCommand,
)
from typing_extensions import Annotated

from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper(
    name="archive",
    help="Commands for managing archives.",
    add_completion=True,
    rich_markup_mode=None,
)


@app.command(name="discover-and-bind")
async def discover_and_bind(
    paths: Annotated[
        List[str],
        typer.Argument(
            ...,
            help="Paths to search for split archives.",
        ),
    ],
):
    """
    Discovers and binds split archives.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    cmd = DiscoverAndBindCommand(paths=paths)
    await target_world.dispatch_command(cmd).get_all_results()
    typer.secho("Discovery and binding process complete.", fg=typer.colors.GREEN)


@app.command(name="create-master")
async def create_master_archive(
    name: Annotated[str, typer.Argument(..., help="The name for the master archive.")],
    part_entity_ids: Annotated[List[int], typer.Argument(..., help="The IDs of the member entities.")],
):
    """
    Creates a master archive from a list of member entities.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    cmd = CreateMasterArchiveCommand(name=name, part_entity_ids=part_entity_ids)
    await target_world.dispatch_command(cmd).get_all_results()
    typer.secho("Master archive creation complete.", fg=typer.colors.GREEN)


@app.command(name="unbind-split")
async def unbind_split_archive(
    entity_id: Annotated[int, typer.Argument(..., help="The ID of the master archive entity.")],
):
    """
    Unbinds a split archive, detaching its members.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    cmd = UnbindSplitArchiveCommand(master_entity_id=entity_id)
    await target_world.dispatch_command(cmd).get_all_results()
    typer.secho("Unbinding process complete.", fg=typer.colors.GREEN)
