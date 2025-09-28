from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import magic
import typer
from dam_archive.commands import (
    ClearArchiveComponentsCommand,
    CreateMasterArchiveCommand,
    DiscoverAndBindCommand,
    UnbindSplitArchiveCommand,
)
from dam_archive.main import open_archive
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


@app.command(name="benchmark")
async def benchmark_archive(
    file_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to the archive file.",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    passwords: Annotated[
        Optional[List[str]],
        typer.Option(
            "--password",
            "-p",
            help="Password to use for the archive. Can be specified multiple times.",
        ),
    ] = None,
):
    """
    Benchmarks the performance of iterating through an archive file.
    """
    typer.echo(f"Benchmarking archive: {file_path}")

    start_time = time.monotonic()
    total_files = 0
    total_size = 0

    passwords_to_try: List[Optional[str]] = [None]
    if passwords:
        passwords_to_try.extend(passwords)

    archive = None

    try:
        mime_type = magic.from_file(str(file_path), mime=True)  # type: ignore
    except Exception as e:
        typer.secho(f"Could not determine mime type for file: {file_path}. Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        with open(file_path, "rb") as f:
            for pwd in passwords_to_try:
                try:
                    f.seek(0)
                    archive = await open_archive(f, mime_type, pwd)
                    if archive:
                        typer.echo(f"Successfully opened archive with password: {'yes' if pwd else 'no'}")
                        break
                except Exception:
                    continue

            if not archive:
                typer.secho(
                    "Could not open archive. It might be password protected or the format is not supported.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

            for member_info, member_stream in archive.iter_files():
                with member_stream:
                    # Read the whole stream to simulate processing
                    member_stream.read()
                    total_files += 1
                    total_size += member_info.size

    except Exception as e:
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    end_time = time.monotonic()
    elapsed_time = end_time - start_time

    if elapsed_time == 0:
        elapsed_time = 1e-9  # Avoid division by zero

    total_size_mb = total_size / (1024 * 1024)
    files_per_second = total_files / elapsed_time
    mb_per_second = total_size_mb / elapsed_time

    typer.echo("\n--- Benchmark Results ---")
    typer.echo(f"Total files: {total_files}")
    typer.echo(f"Total size: {total_size_mb:.2f} MB")
    typer.echo(f"Elapsed time: {elapsed_time:.2f} seconds")
    typer.echo(f"Files per second: {files_per_second:.2f}")
    typer.echo(f"Processing speed: {mb_per_second:.2f} MB/s")
