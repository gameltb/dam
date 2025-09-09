import datetime
from pathlib import Path
from typing import List

import typer
from dam_fs.commands import (
    FindEntityByFilePropertiesCommand,
    RegisterLocalFileCommand,
    StoreAssetsCommand,
)
from tqdm import tqdm
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
    """
    target_world = get_world()
    if not target_world:
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

    total_size = sum(p.stat().st_size for p in files_to_process)

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Registering assets") as pbar:
        for file_path in files_to_process:
            pbar.set_postfix_str(file_path.name)
            try:
                mod_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime, tz=datetime.timezone.utc)
                pre_check_cmd = FindEntityByFilePropertiesCommand(
                    file_path=file_path.as_uri(), file_modified_at=mod_time
                )
                cmd_result = await target_world.dispatch_command(pre_check_cmd)
                existing_entity_id = cmd_result.get_one_value()

                if existing_entity_id:
                    skipped_count += 1
                else:
                    register_cmd = RegisterLocalFileCommand(file_path=file_path)
                    await target_world.dispatch_command(register_cmd)
                    success_count += 1

            except Exception as e:
                error_count += 1
                tqdm.write(f"Error processing file {file_path.name}: {e}")
            pbar.update(file_path.stat().st_size)

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
