import datetime
from pathlib import Path
from typing import List

import typer
from dam_fs.commands import (
    FindEntityByFilePropertiesCommand,
    RegisterLocalFileCommand,
    StoreAssetsCommand,
)
from typing_extensions import Annotated

from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper()


@app.command(name="add")
async def add_assets(
    paths: Annotated[
        List[Path],
        typer.Argument(
            ...,
            help="Path to the asset file or directory of asset files.",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    recursive: Annotated[
        bool,
        typer.Option("-r", "--recursive", help="Process directory recursively."),
    ] = False,
):
    """
    Registers one or more local assets with the DAM.

    This command finds files, calculates their hashes, and creates or updates
    asset entities in the DAM. It does NOT copy the files to the DAM's
    internal storage.
    """
    target_world = get_world()
    if not target_world:
        # The main callback in cli.py should handle the error message.
        raise typer.Exit(code=1)

    typer.echo("Starting asset registration process...")

    files_to_process: List[Path] = []
    for path in paths:
        if path.is_file():
            files_to_process.append(path)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            files_to_process.extend(p for p in path.glob(pattern) if p.is_file())

    if not files_to_process:
        typer.secho("No files found to process.", fg=typer.colors.YELLOW)
        return

    typer.echo(f"Found {len(files_to_process)} file(s) to process.")

    success_count = 0
    skipped_count = 0
    error_count = 0

    for file_path in files_to_process:
        try:
            typer.echo(f"Processing: {file_path.name}")

            # 1. Pre-check based on path and mod time
            mod_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime, tz=datetime.timezone.utc)
            pre_check_cmd = FindEntityByFilePropertiesCommand(file_path=file_path.as_uri(), file_modified_at=mod_time)
            cmd_result = await target_world.dispatch_command(pre_check_cmd)
            existing_entity_id = cmd_result.get_one_value()

            if existing_entity_id:
                # For now, we assume if it's found by path and mod time, its hashes are correct.
                # A more robust check could be added later if needed.
                typer.secho(
                    f"  Skipping '{file_path.name}', up-to-date entity {existing_entity_id} already exists.",
                    fg=typer.colors.YELLOW,
                )
                skipped_count += 1
                continue

            # 2. If pre-check fails, register the file (which includes hash check)
            register_cmd = RegisterLocalFileCommand(file_path=file_path)
            register_result = await target_world.dispatch_command(register_cmd)
            new_entity_id = register_result.get_one_value()
            typer.secho(
                f"  Successfully registered '{file_path.name}' as entity {new_entity_id}.", fg=typer.colors.GREEN
            )
            success_count += 1

        except Exception as e:
            typer.secho(f"  Error processing file {file_path.name}: {e}", fg=typer.colors.RED)
            error_count += 1

    typer.echo("\n--- Summary ---")
    typer.echo(f"Successfully registered: {success_count}")
    typer.echo(f"Skipped (up-to-date): {skipped_count}")
    typer.echo(f"Errors: {error_count}")
    typer.secho("Asset registration complete.", fg=typer.colors.GREEN)


@app.command(name="store")
async def store_assets(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            "-q",
            help="A query to select assets to store. Defaults to all local files not in storage.",
        ),
    ] = "local_not_stored",
):
    """
    Copies registered local assets into the DAM's content-addressable storage.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    typer.echo(f"Storing assets for query: '{query}'...")

    store_cmd = StoreAssetsCommand(query=query)
    await target_world.dispatch_command(store_cmd)

    typer.secho("Asset storage process complete.", fg=typer.colors.GREEN)
