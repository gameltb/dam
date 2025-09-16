from typing import List, Optional

import typer
from dam.models.conceptual.mime_type_concept_component import MimeTypeConceptComponent
from dam.models.core import Entity
from dam.models.metadata.content_mime_type_component import ContentMimeTypeComponent
from dam.system_events.progress import (
    ProgressCompleted,
    ProgressError,
    ProgressStarted,
    ProgressUpdate,
)
from dam_archive.commands import IngestArchiveMembersCommand
from dam_archive.exceptions import PasswordRequiredError
from dam_archive.models import ArchiveInfoComponent
from dam_archive.registry import MIME_TYPE_HANDLERS
from sqlalchemy import select
from tqdm.asyncio import tqdm
from typing_extensions import Annotated

from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper()
run_app = AsyncTyper(name="run", help="Run specific processing systems.")
app.add_typer(run_app, name="run")


@run_app.command(name="process-archives")
async def process_archives(
    passwords: Annotated[
        Optional[List[str]],
        typer.Option(
            "--password",
            "-p",
            help="Password to try for encrypted archives. Can be specified multiple times.",
        ),
    ] = None,
):
    """
    Finds and processes archive files (e.g., .zip, .rar) in the DAM.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    known_passwords = passwords or []

    typer.echo("Finding archives to process...")

    async with target_world.db_session_maker() as session:
        stmt = (
            select(Entity.id)
            .join(ContentMimeTypeComponent, Entity.id == ContentMimeTypeComponent.entity_id)
            .join(
                MimeTypeConceptComponent,
                ContentMimeTypeComponent.mime_type_concept_id == MimeTypeConceptComponent.id,
            )
            .outerjoin(ArchiveInfoComponent, Entity.id == ArchiveInfoComponent.entity_id)
            .where(MimeTypeConceptComponent.mime_type.in_(MIME_TYPE_HANDLERS.keys()))
            .where(ArchiveInfoComponent.id.is_(None))
        )
        result = await session.execute(stmt)
        archives_to_process = result.scalars().all()

    if not archives_to_process:
        typer.secho("No new archives found to process.", fg=typer.colors.GREEN)
        return

    typer.echo(f"Found {len(archives_to_process)} archive(s) to process.")

    for entity_id in archives_to_process:
        typer.echo(f"Processing archive entity {entity_id}...")

        processing_failed = False
        while not processing_failed:
            async with target_world.transaction():
                cmd = IngestArchiveMembersCommand(entity_id=entity_id, passwords=known_passwords)
                pbar: Optional[tqdm] = None
                try:
                    stream = target_world.dispatch_command(cmd)
                    async for event in stream:
                        match event:
                            case ProgressStarted():
                                pass
                            case ProgressUpdate(total, current, message):
                                if pbar is None and total is not None:
                                    pbar = tqdm(
                                        total=total,
                                        unit="B",
                                        unit_scale=True,
                                        desc="Extracting members",
                                        smoothing=0,
                                    )
                                if pbar is not None and current is not None:
                                    pbar.update(current - pbar.n)
                                if message and pbar:
                                    pbar.set_postfix_str(message)
                            case ProgressError(exception, _):
                                if isinstance(exception, PasswordRequiredError):
                                    typer.secho(f"  Password required for archive {entity_id}.", fg=typer.colors.YELLOW)
                                    new_password = typer.prompt(
                                        "Enter password (or press Enter to skip)", default="", show_default=False
                                    )
                                    if new_password and new_password not in known_passwords:
                                        known_passwords.append(new_password)
                                        # Break inner loop to retry with new password
                                        break
                                    else:
                                        typer.secho(f"  Skipping archive {entity_id}.", fg=typer.colors.YELLOW)
                                        processing_failed = True
                                else:
                                    typer.secho(
                                        f"  An error occurred while processing archive {entity_id}: {exception}",
                                        fg=typer.colors.RED,
                                    )
                                    processing_failed = True
                            case ProgressCompleted(message):
                                typer.secho(
                                    f"  Successfully processed archive {entity_id}. {message or ''}",
                                    fg=typer.colors.GREEN,
                                )
                            case _:
                                pass

                        if processing_failed:
                            break  # Stop processing events for this archive
                    else:
                        # This 'else' belongs to the 'for' loop, executed when the loop finishes without 'break'
                        break  # Exit the 'while' loop on successful completion
                finally:
                    if pbar:
                        pbar.close()

    typer.echo("Archive processing complete.")
