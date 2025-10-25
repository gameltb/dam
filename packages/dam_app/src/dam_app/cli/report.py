"""CLI commands for generating reports."""

import csv
import hashlib
import math
import re
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.models.hashes.content_hash_sha256_component import ContentHashSHA256Component
from dam_archive.models import ArchiveMemberComponent
from dam_fs.models.file_location_component import FileLocationComponent
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from dam_app.functions.report import (
    DeletePlanRow,
    DuplicateRow,
    create_delete_plan,
    get_duplicates_report,
)
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


def _write_csv_report(csv_path: Path, duplicates: Sequence[DuplicateRow]):
    """Write the duplicate file report to a CSV file."""
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Entity ID",
                "SHA256",
                "Size (bytes)",
                "Size (HR)",
                "Size on Disk (bytes)",
                "Size on Disk (HR)",
                "Locations",
                "Wasted Space (bytes)",
                "Path",
                "Type",
            ]
        )
        for duplicate in duplicates:
            entity_id = duplicate.entity_id
            total_locations = duplicate.total_locations
            size_bytes = duplicate.file_size_bytes or 0
            size_on_disk_bytes = duplicate.size_on_disk
            hash_hex = duplicate.hash_value.hex()
            wasted_space = size_bytes * (total_locations - 1)
            path = duplicate.path
            type = duplicate.type

            writer.writerow(
                [
                    str(entity_id),
                    hash_hex,
                    str(size_bytes),
                    _human_readable_size(size_bytes),
                    str(size_on_disk_bytes),
                    _human_readable_size(size_on_disk_bytes),
                    str(total_locations),
                    str(wasted_space),
                    path,
                    type,
                ]
            )


def _print_rich_report(console: Console, duplicates: Sequence[DuplicateRow]):
    """Print the duplicate file report to the console using rich."""
    table = Table(title="Duplicate Files Report")
    table.add_column("Entity ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("SHA256", style="magenta")
    table.add_column("Size (bytes)", justify="right", style="green")
    table.add_column("Size (HR)", justify="right", style="green")
    table.add_column("Size on Disk (bytes)", justify="right", style="green")
    table.add_column("Size on Disk (HR)", justify="right", style="green")
    table.add_column("Locations", justify="right", style="red")
    table.add_column("Wasted Space (bytes)", justify="right", style="yellow")
    table.add_column("Path", style="blue")
    table.add_column("Type", style="blue")

    total_wasted_space = 0
    processed_entities: set[int] = set()
    for duplicate in duplicates:
        entity_id = duplicate.entity_id
        total_locations = duplicate.total_locations
        size_bytes = duplicate.file_size_bytes or 0
        size_on_disk_bytes = duplicate.size_on_disk
        hash_hex = duplicate.hash_value.hex()
        if entity_id not in processed_entities:
            wasted_space = size_bytes * (total_locations - 1)
            total_wasted_space += wasted_space
            processed_entities.add(entity_id)
        else:
            wasted_space = 0

        path = duplicate.path
        type = duplicate.type

        table.add_row(
            str(entity_id),
            f"{hash_hex[:16]}...",
            str(size_bytes),
            _human_readable_size(size_bytes),
            str(size_on_disk_bytes),
            _human_readable_size(size_on_disk_bytes),
            str(total_locations),
            str(wasted_space),
            path,
            type,
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

        if csv_path:
            _write_csv_report(csv_path, duplicates)
            console.print(f"Report written to {csv_path}")
        else:
            _print_rich_report(console, duplicates)


def _write_delete_plan_csv_report(csv_path: Path, delete_plan: Sequence[DeletePlanRow]):
    """Write the delete plan report to a CSV file."""
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "source_path",
                "target_path",
                "hash",
                "size",
                "details",
            ]
        )
        for row in delete_plan:
            writer.writerow(
                [
                    row.source_path,
                    row.target_path,
                    row.hash,
                    str(row.size),
                    row.details,
                ]
            )


@app.command("create-delete-report")
async def create_delete_report(
    csv_path: Annotated[
        Path,
        typer.Option(
            "--csv-path",
            "-c",
            help="Path to the CSV file to write the report to.",
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    keep_patterns: Annotated[
        list[str] | None,
        typer.Option(
            "--keep",
            "-k",
            help="File name patterns to keep. Files matching these patterns will not be deleted.",
        ),
    ] = None,
    delete_patterns: Annotated[
        list[str] | None,
        typer.Option(
            "--delete",
            "-d",
            help="File name patterns to delete. Files matching these patterns will be prioritized for deletion.",
        ),
    ] = None,
    min_size: Annotated[
        int,
        typer.Option(
            "--min-size",
            "-m",
            help="Minimum size in MB for reporting duplicates in archives.",
        ),
    ] = 100,
):
    """Scan for duplicate files and generate a deletion plan."""
    world = get_world()
    if not world:
        return

    console = Console()
    min_size_bytes = min_size * 1024 * 1024

    async with world.get_context(WorldTransaction)() as transaction:
        session = transaction.session
        delete_plan = await create_delete_plan(
            session=session,
            min_size_bytes=min_size_bytes,
            keep_patterns=keep_patterns,
            delete_patterns=delete_patterns,
        )

        if not delete_plan:
            console.print("No duplicate files found to delete.")
            return

        _write_delete_plan_csv_report(csv_path, delete_plan)
        console.print(f"Delete plan report written to {csv_path}")


def _calculate_file_sha256(file_path: Path) -> str:
    """Calculate the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


async def verify_archive_members(world: World, archive_path: Path, console: Console) -> bool:
    """Verify that the members of an archive match the hashes in the database."""
    try:
        async with world.get_context(WorldTransaction)() as transaction:
            # Get the entity ID for the archive path
            archive_entity_id_query = select(FileLocationComponent.entity_id).where(
                FileLocationComponent.url == f"file://{archive_path.resolve()}"
            )
            archive_entity_id = (await transaction.session.execute(archive_entity_id_query)).scalar_one_or_none()
            if not archive_entity_id:
                console.print(f"[red]Could not find entity for archive: {archive_path}[/red]")
                return False

            # Get the expected hashes of all members from the database
            member_hashes_query = (
                select(ArchiveMemberComponent.path_in_archive, ContentHashSHA256Component.hash_value)
                .join(
                    ContentHashSHA256Component,
                    ContentHashSHA256Component.entity_id == ArchiveMemberComponent.entity_id,
                )
                .where(ArchiveMemberComponent.archive_entity_id == archive_entity_id)
            )
            expected_hashes = {
                path: hash_value.hex() for path, hash_value in (await transaction.session.execute(member_hashes_query))
            }

        if not expected_hashes:
            console.print(f"[yellow]No members found in database for archive: {archive_path}[/yellow]")
            # If the archive is empty and we expect it to be, it's valid.
            return True

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            for member_path, expected_hash in expected_hashes.items():
                actual_file_path = temp_path / member_path
                if not actual_file_path.is_file():
                    console.print(f"[red]Member file not found in archive: {member_path}[/red]")
                    return False

                actual_hash = _calculate_file_sha256(actual_file_path)
                if actual_hash != expected_hash:
                    console.print(f"[red]Hash mismatch for member {member_path} in {archive_path}[/red]")
                    return False

        return True

    except Exception as e:
        console.print(f"[red]Error verifying archive {archive_path}: {e}[/red]")
        return False


@app.command("execute-delete-report")
async def execute_delete_report(  # noqa: PLR0912
    csv_path: Annotated[
        Path,
        typer.Option(
            "--csv-path",
            "-c",
            help="Path to the CSV file with the delete plan.",
            dir_okay=False,
            resolve_path=True,
            exists=True,
        ),
    ],
):
    """Execute the delete plan from a CSV report."""
    console = Console()
    world = get_world()
    if not world:
        console.print("[red]Error: Could not get world object.[/red]")
        return

    deleted_count = 0
    skipped_count = 0
    extracted_count = 0

    with csv_path.open("r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        if header != ["source_path", "target_path", "hash", "size", "details"]:
            console.print("[red]Error: Invalid CSV file format.[/red]")
            return

        for row in reader:
            target_path_str = row[1]
            expected_hash = row[2]
            details = row[4]

            target_path = Path(target_path_str)

            if not target_path.is_file():
                console.print(f"[yellow]Skipping (not a file): {target_path}[/yellow]")
                skipped_count += 1
                continue

            is_archive = zipfile.is_zipfile(target_path)
            is_fully_duplicate_archive = "All members are duplicates" in details

            if is_archive:
                if is_fully_duplicate_archive:
                    if await verify_archive_members(world, target_path, console):
                        try:
                            target_path.unlink()
                            console.print(f"[green]Deleted archive: {target_path}[/green]")
                            deleted_count += 1
                        except OSError as e:
                            console.print(f"[red]Error deleting archive {target_path}: {e}[/red]")
                            skipped_count += 1
                    else:
                        console.print(f"[yellow]Skipping archive (verification failed): {target_path}[/yellow]")
                        skipped_count += 1
                else:
                    # Partial duplicate archive handling
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)
                            with zipfile.ZipFile(target_path, "r") as zip_ref:
                                zip_ref.extractall(temp_path)

                            # Identify non-duplicate members
                            duplicate_members = set(re.findall(r"'[^']+\s->\s([^']+)'.*?is a duplicate", details))
                            all_members = {str(p.relative_to(temp_path)) for p in temp_path.rglob("*") if p.is_file()}
                            non_duplicate_members = all_members - duplicate_members

                            if non_duplicate_members:
                                extract_dir = target_path.with_suffix("")
                                extract_dir.mkdir(exist_ok=True)
                                console.print(f"Extracting non-duplicate files to {extract_dir}")
                                for member in non_duplicate_members:
                                    source = temp_path / member
                                    destination = extract_dir / member
                                    destination.parent.mkdir(parents=True, exist_ok=True)
                                    source.rename(destination)
                                    console.print(f"  - Extracted: {destination}")
                                    extracted_count += 1

                        target_path.unlink()
                        console.print(f"[green]Deleted archive after partial extraction: {target_path}[/green]")
                        deleted_count += 1

                    except (zipfile.BadZipFile, OSError) as e:
                        console.print(f"[red]Error processing archive {target_path}: {e}[/red]")
                        skipped_count += 1
            else:
                # Regular file deletion
                actual_hash = _calculate_file_sha256(target_path)
                if actual_hash == expected_hash:
                    try:
                        target_path.unlink()
                        console.print(f"[green]Deleted: {target_path}[/green]")
                        deleted_count += 1
                    except OSError as e:
                        console.print(f"[red]Error deleting {target_path}: {e}[/red]")
                        skipped_count += 1
                else:
                    console.print(f"[yellow]Skipping (hash mismatch): {target_path}[/yellow]")
                    skipped_count += 1

    console.print("\n--- Summary ---")
    console.print(f"Files deleted: {deleted_count}")
    console.print(f"Files extracted: {extracted_count}")
    console.print(f"Files skipped: {skipped_count}")
