from typing import Optional

import typer
from dam.core.transaction import WorldTransaction
from dam.events.asset_events import AssetReadyForMetadataExtractionEvent
from dam.models.core.entity import Entity
from dam.models.metadata.content_mime_type_component import ContentMimeTypeComponent
from dam.system_events.progress import (
    ProgressCompleted,
    ProgressError,
    ProgressStarted,
    ProgressUpdate,
)
from dam_archive.commands import IngestArchiveCommand
from rich.progress import Progress
from typing_extensions import Annotated

from dam_app.commands import AutoTagEntityCommand
from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper()


@app.command(name="run-metadata-extraction")
async def run_metadata_extraction(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            "-q",
            help="A query to select assets to process. Defaults to all assets.",
        ),
    ] = "*",
):
    """
    Runs the metadata extraction process on a selection of assets.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    typer.echo(f"Running metadata extraction for assets matching query: '{query}'...")

    # This is a simplified example. A real implementation would query entities.
    # For now, we'll simulate finding some entities.
    entity_ids = [1, 2, 3]  # Dummy entity IDs

    event = AssetReadyForMetadataExtractionEvent(entity_ids=entity_ids)
    await target_world.dispatch_event(event)

    typer.secho("Metadata extraction process complete.", fg=typer.colors.GREEN)


@app.command(name="run-auto-tagging")
async def run_auto_tagging(
    entity_id: Annotated[int, typer.Argument(help="The ID of the entity to tag.")],
):
    """
    Runs the auto-tagging process on a single entity.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    typer.echo(f"Running auto-tagging for entity: {entity_id}...")

    async with target_world.get_context(WorldTransaction)() as tx:
        session = tx.session
        entity = await session.get(Entity, entity_id)
        if not entity:
            typer.secho(f"Entity with ID {entity_id} not found.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        cmd = AutoTagEntityCommand(entity=entity)
        await target_world.dispatch_command(cmd).get_all_results()

    typer.secho("Auto-tagging process complete.", fg=typer.colors.GREEN)


@app.command(name="ingest-archive")
async def ingest_archive(
    entity_id: Annotated[int, typer.Argument(help="The ID of the archive entity to ingest.")],
    password: Annotated[Optional[str], typer.Option("--password", "-p", help="Password for the archive.")] = None,
):
    """
    Ingests the members of an archive asset into the DAM.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    async with target_world.get_context(WorldTransaction)() as tx:
        session = tx.session
        mime_type_comp = await session.get(ContentMimeTypeComponent, entity_id)
        mime_type = ""
        if mime_type_comp and mime_type_comp.mime_type_concept:
            mime_type = mime_type_comp.mime_type_concept.mime_type

        known_passwords = [password] if password else []

        with Progress() as progress:
            task = progress.add_task(f"[cyan]Ingesting archive {entity_id}...", total=None)

            if mime_type and mime_type.startswith("application/"):
                cmd = IngestArchiveCommand(entity_id=entity_id, passwords=known_passwords)
                async for event in target_world.dispatch_command(cmd):
                    if isinstance(event, ProgressStarted):
                        progress.update(task, total=100, completed=0)
                    elif isinstance(event, ProgressUpdate):
                        progress.update(
                            task, total=event.total, completed=event.current, description=f"[cyan]{event.message}"
                        )
                    elif isinstance(event, ProgressCompleted):
                        progress.update(task, completed=progress.tasks[task].total, description="[green]Complete")
                    elif isinstance(event, ProgressError):
                        progress.update(task, description=f"[red]Error: {event.message}")
                        break
            else:
                progress.update(task, description="[yellow]Not an archive, skipping.")

    typer.secho("Archive ingestion process complete.", fg=typer.colors.GREEN)
