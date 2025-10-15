"""CLI commands for generating reports."""

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from dam.core.transaction import WorldTransaction
from rich.console import Console
from rich.table import Table

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
                "Compressed Size (bytes)",
                "Compressed Size (HR)",
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
            compressed_size_bytes = duplicate.compressed_size_bytes
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
                    str(compressed_size_bytes),
                    _human_readable_size(compressed_size_bytes),
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
    table.add_column("Compressed Size (bytes)", justify="right", style="green")
    table.add_column("Compressed Size (HR)", justify="right", style="green")
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
        compressed_size_bytes = duplicate.compressed_size_bytes
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
            str(compressed_size_bytes),
            _human_readable_size(compressed_size_bytes),
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
