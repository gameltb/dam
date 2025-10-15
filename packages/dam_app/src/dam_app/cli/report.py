"""CLI commands for generating reports."""

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from dam.core.transaction import WorldTransaction
from dam_archive.models import ArchiveMemberComponent
from dam_fs.models.file_location_component import FileLocationComponent
from dam_psp.models import CsoParentIsoComponent
from rich.console import Console
from rich.table import Table
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dam_app.functions.report import DuplicateRow, get_duplicates_report
from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper(
    name="report",
    help="Commands for generating reports.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="markdown",
)


def _human_readable_size(size_bytes: int | None) -> str:
    """Return a human-readable size string."""
    if size_bytes is None:
        return ""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = math.floor(math.log(size_bytes, 1024))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


async def _get_paths(session: AsyncSession, entity_id: int) -> list[str]:
    """Get all paths for a given entity."""
    paths: list[str] = []
    file_locations_result = await session.execute(
        select(FileLocationComponent).where(FileLocationComponent.entity_id == entity_id)
    )
    file_locations = file_locations_result.scalars().all()
    archive_members_result = await session.execute(
        select(ArchiveMemberComponent).where(ArchiveMemberComponent.entity_id == entity_id)
    )
    archive_members = archive_members_result.scalars().all()
    cso_parents_result = await session.execute(
        select(CsoParentIsoComponent).where(CsoParentIsoComponent.entity_id == entity_id)
    )
    cso_parents = cso_parents_result.scalars().all()

    for loc in file_locations:
        paths.append(f"Filesystem: {loc.url}")

    for member in archive_members:
        archive_file_locations_result = await session.execute(
            select(FileLocationComponent).where(FileLocationComponent.entity_id == member.archive_entity_id)
        )
        archive_file_locations = archive_file_locations_result.scalars().all()
        for archive_loc in archive_file_locations:
            paths.append(f"Archive: {archive_loc.url} -> {member.path_in_archive}")

    for cso in cso_parents:
        cso_file_locations_result = await session.execute(
            select(FileLocationComponent).where(FileLocationComponent.entity_id == cso.cso_entity_id)
        )
        cso_file_locations = cso_file_locations_result.scalars().all()
        for cso_loc in cso_file_locations:
            paths.append(f"CSO: {cso_loc.url}")
    return paths


def _write_csv_report(csv_path: Path, duplicates: Sequence[DuplicateRow], all_paths: dict[int, list[str]]):
    """Write the duplicate file report to a CSV file."""
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Entity ID",
                "SHA256",
                "Size (bytes)",
                "Size (HR)",
                "Locations",
                "Wasted Space (bytes)",
                "Paths",
            ]
        )
        for duplicate in duplicates:
            entity_id = duplicate.entity_id
            total_locations = duplicate.total_locations
            size_bytes = duplicate.file_size_bytes or 0
            hash_hex = duplicate.hash_value.hex()
            wasted_space = size_bytes * (total_locations - 1)
            paths = all_paths.get(entity_id, [])

            writer.writerow(
                [
                    str(entity_id),
                    f"{hash_hex[:16]}...",
                    str(size_bytes),
                    _human_readable_size(size_bytes),
                    str(total_locations),
                    str(wasted_space),
                    "\n".join(paths),
                ]
            )


def _print_rich_report(console: Console, duplicates: Sequence[DuplicateRow], all_paths: dict[int, list[str]]):
    """Print the duplicate file report to the console using rich."""
    table = Table(title="Duplicate Files Report")
    table.add_column("Entity ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("SHA256", style="magenta")
    table.add_column("Size (bytes)", justify="right", style="green")
    table.add_column("Size (HR)", justify="right", style="green")
    table.add_column("Locations", justify="right", style="red")
    table.add_column("Wasted Space (bytes)", justify="right", style="yellow")
    table.add_column("Paths", style="blue")

    total_wasted_space = 0
    for duplicate in duplicates:
        entity_id = duplicate.entity_id
        total_locations = duplicate.total_locations
        size_bytes = duplicate.file_size_bytes or 0
        hash_hex = duplicate.hash_value.hex()
        wasted_space = size_bytes * (total_locations - 1)
        total_wasted_space += wasted_space
        paths = all_paths.get(entity_id, [])

        table.add_row(
            str(entity_id),
            f"{hash_hex[:16]}...",
            str(size_bytes),
            _human_readable_size(size_bytes),
            str(total_locations),
            str(wasted_space),
            "\n".join(paths),
        )
    console.print(table)
    console.print(f"\nTotal wasted space: {total_wasted_space} bytes")


@app.command("duplicates")
async def report_duplicates(
    csv_path: Annotated[
        Path | None,
        typer.Option(
            "--csv",
            "-c",
            help="Path to the CSV file to write the report to.",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            "-p",
            help="Path to filter the report by.",
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
):
    """Report on duplicate files in the world."""
    world = get_world()
    if not world:
        return

    console = Console()

    async with world.get_context(WorldTransaction)() as transaction:
        session = transaction.session
        duplicates = await get_duplicates_report(session, path)

        if not duplicates:
            console.print("No duplicate files found.")
            return

        all_paths = {}
        for duplicate in duplicates:
            all_paths[duplicate.entity_id] = await _get_paths(session, duplicate.entity_id)

        if csv_path:
            _write_csv_report(csv_path, duplicates, all_paths)
            console.print(f"Report written to {csv_path}")
        else:
            _print_rich_report(console, duplicates, all_paths)
