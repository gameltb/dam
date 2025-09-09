from typing import List, Optional

import typer
from dam.models.core import Entity
from dam_archive.commands import ExtractArchiveCommand
from dam_archive.exceptions import PasswordRequiredError
from dam_archive.models import ArchiveInfoComponent
from dam_fs.models import FilePropertiesComponent
from sqlalchemy import and_, select
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
        # Find entities that have a file-like name but no ArchiveInfoComponent
        stmt = (
            select(Entity)
            .join(FilePropertiesComponent)
            .outerjoin(ArchiveInfoComponent)
            .where(
                and_(
                    FilePropertiesComponent.original_filename.op("~*")(".(zip|rar|7z)$"),
                    ArchiveInfoComponent.id.is_(None),
                )
            )
        )
        result = await session.execute(stmt)
        archives_to_process = result.scalars().all()

    if not archives_to_process:
        typer.secho("No archives found to process.", fg=typer.colors.GREEN)
        return

    typer.echo(f"Found {len(archives_to_process)} archive(s) to process.")

    for entity in archives_to_process:
        typer.echo(f"Processing archive entity {entity.id}...")

        while True:
            try:
                cmd = ExtractArchiveCommand(entity_id=entity.id, passwords=known_passwords)
                await target_world.dispatch_command(cmd)
                typer.secho(f"  Successfully processed archive {entity.id}", fg=typer.colors.GREEN)
                break  # Success, move to next archive
            except PasswordRequiredError:
                typer.secho(f"  Password required for archive {entity.id}.", fg=typer.colors.YELLOW)
                new_password = typer.prompt("Enter password (or press Enter to skip)", default="", show_default=False)
                if not new_password:
                    typer.secho(f"  Skipping archive {entity.id}.", fg=typer.colors.YELLOW)
                    break  # Skip this archive

                if new_password not in known_passwords:
                    known_passwords.append(new_password)
                # Loop will continue and try again with the new password
            except Exception as e:
                typer.secho(
                    f"  An unexpected error occurred while processing archive {entity.id}: {e}", fg=typer.colors.RED
                )
                break  # Unrecoverable error for this archive

    typer.echo("Archive processing complete.")
