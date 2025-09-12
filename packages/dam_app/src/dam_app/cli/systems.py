from typing import Any, List, Optional

import typer
from dam.commands import GetAssetFilenamesCommand
from dam.models.core import Entity
from dam_archive.commands import ExtractArchiveMembersCommand
from dam_archive.exceptions import PasswordRequiredError
from dam_archive.models import ArchiveInfoComponent
from sqlalchemy import select
from tqdm import tqdm
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
        # Find entities that could be archives but haven't been processed
        stmt = (
            select(Entity.id)
            .select_from(Entity)
            .outerjoin(ArchiveInfoComponent, Entity.id == ArchiveInfoComponent.entity_id)
            .where(ArchiveInfoComponent.id.is_(None))
        )
        result = await session.execute(stmt)
        candidate_ids = result.scalars().all()

    archives_to_process: List[int] = []
    for entity_id in candidate_ids:
        cmd = GetAssetFilenamesCommand(entity_id=entity_id)
        cmd_result = await target_world.dispatch_command(cmd)
        for filename in cmd_result.iter_ok_values_flat():
            if filename.lower().endswith((".zip", ".rar", ".7z")):
                archives_to_process.append(entity_id)
                break

    if not archives_to_process:
        typer.secho("No new archives found to process.", fg=typer.colors.GREEN)
        return

    typer.echo(f"Found {len(archives_to_process)} archive(s) to process.")

    for entity_id in archives_to_process:
        typer.echo(f"Processing archive entity {entity_id}...")

        pbar: Optional[tqdm[Any]] = None

        def init_progress(total_size: int):
            nonlocal pbar
            pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Extracting members")

        def update_progress(size: int):
            if pbar:
                pbar.update(size)

        def error_handler(member_name: str, e: Exception) -> bool:
            typer.secho(f"  Failed to process member '{member_name}': {e}", fg=typer.colors.RED)
            return typer.confirm("Do you want to continue with other files?")

        while True:
            try:
                cmd = ExtractArchiveMembersCommand(
                    entity_id=entity_id,
                    passwords=known_passwords,
                    init_progress_callback=init_progress,
                    update_progress_callback=update_progress,
                    error_callback=error_handler,
                )
                await target_world.dispatch_command(cmd)
                if pbar:
                    pbar.close()
                typer.secho(f"  Successfully processed archive {entity_id}", fg=typer.colors.GREEN)
                break
            except PasswordRequiredError:
                if pbar:
                    pbar.close()
                typer.secho(f"  Password required for archive {entity_id}.", fg=typer.colors.YELLOW)
                new_password = typer.prompt("Enter password (or press Enter to skip)", default="", show_default=False)
                if not new_password:
                    typer.secho(f"  Skipping archive {entity_id}.", fg=typer.colors.YELLOW)
                    break
                if new_password not in known_passwords:
                    known_passwords.append(new_password)
            except Exception as e:
                if pbar:
                    pbar.close()
                typer.secho(
                    f"  An unexpected error occurred while processing archive {entity_id}: {e}", fg=typer.colors.RED
                )
                break

    typer.echo("Archive processing complete.")
