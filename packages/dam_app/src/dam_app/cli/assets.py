from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from dam.commands.analysis_commands import AutoSetMimeTypeCommand
from dam.commands.asset_commands import (
    GetAssetFilenamesCommand,
    GetMimeTypeCommand,
    SetMimeTypeCommand,
)
from dam.functions import ecs_functions as dam_ecs_functions
from dam.system_events.entity_events import NewEntityCreatedEvent
from dam.system_events.progress import (
    ProgressCompleted,
    ProgressError,
    ProgressStarted,
    ProgressUpdate,
)
from dam_archive.commands import IngestArchiveCommand
from dam_fs.commands import (
    FindEntityByFilePropertiesCommand,
    RegisterLocalFileCommand,
    StoreAssetsCommand,
)
from dam_psp.commands import ExtractPSPMetadataCommand
from rich.console import Console
from rich.traceback import Traceback
from tqdm import tqdm
from typing_extensions import Annotated

from dam_app.commands import ExtractExifMetadataCommand
from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper()

COMMAND_MAP = {
    "ExtractExifMetadataCommand": ExtractExifMetadataCommand,
    "IngestArchiveCommand": IngestArchiveCommand,
    "ExtractPSPMetadataCommand": ExtractPSPMetadataCommand,
}


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
    process: Annotated[
        Optional[List[str]],
        typer.Option(
            "--process",
            "-p",
            help="Specify a command to run for a given MIME type or file extension, e.g., 'image/jpeg:ExtractExifMetadataCommand' or '.zip:IngestArchiveCommand'",
        ),
    ] = None,
    stop_on_error: Annotated[
        bool,
        typer.Option("--stop-on-error/--no-stop-on-error", help="Stop processing if an error occurs."),
    ] = True,
):
    """
    Registers one or more local assets with the DAM.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    # Parse process option
    process_map: Dict[str, List[str]] = {}
    if process:
        for p in process:
            if ":" in p:
                # Handle traditional 'key:Command' format
                try:
                    key, command_name = p.split(":", 1)
                    if key not in process_map:
                        process_map[key] = []
                    process_map[key].append(command_name)
                except ValueError:
                    # This should not happen due to the ':' check, but for safety:
                    typer.secho(
                        f"Invalid format for --process option: '{p}'. Must be 'key:CommandName'",
                        fg=typer.colors.RED,
                    )
                    raise typer.Exit(code=1)
            else:
                # Handle new 'CommandName' format
                command_name = p
                command_class = COMMAND_MAP.get(command_name)
                if not command_class:
                    typer.secho(f"Unknown command '{command_name}' specified.", fg=typer.colors.RED)
                    raise typer.Exit(code=1)

                if not hasattr(command_class, "get_supported_types"):
                    typer.secho(
                        f"Command '{command_name}' does not support automatic type resolution.", fg=typer.colors.YELLOW
                    )
                    continue

                supported_types = command_class.get_supported_types()
                for mime_type in supported_types.get("mimetypes", []):
                    if mime_type not in process_map:
                        process_map[mime_type] = []
                    process_map[mime_type].append(command_name)
                for extension in supported_types.get("extensions", []):
                    if extension not in process_map:
                        process_map[extension] = []
                    process_map[extension].append(command_name)

    async def _process_entity(
        entity_id: int,
        depth: int,
        pbar: tqdm[Any],
        filename: Optional[str] = None,
        stream_provider_from_event: Optional[Any] = None,
    ):
        """Inner function to handle processing of a single entity."""
        if depth >= 10:
            tqdm.write(f"Skipping entity {entity_id} at depth {depth}, limit reached.")
            return

        # 1. Get MIME type and filename
        entity_filename: Optional[str] = filename

        # Auto-set and get MIME type
        await target_world.dispatch_command(
            AutoSetMimeTypeCommand(entity_id=entity_id, stream_provider=stream_provider_from_event)
        ).get_all_results()
        mime_type_str = await target_world.dispatch_command(GetMimeTypeCommand(entity_id=entity_id)).get_one_value()

        # Get filename if not provided
        if not entity_filename:
            filenames = await target_world.dispatch_command(
                GetAssetFilenamesCommand(entity_id=entity_id)
            ).get_one_value()
            if filenames:
                entity_filename = filenames[0]

        pbar.set_postfix_str(f"Processing {entity_id} ({entity_filename or 'No Filename'})")

        # 2. Collect commands based on MIME type and file extension (MIME type has priority)
        commands_to_run: List[str] = []
        if mime_type_str and mime_type_str in process_map:
            commands_to_run.extend(process_map[mime_type_str])
        elif entity_filename:
            ext = Path(entity_filename).suffix.lower()
            if ext and ext in process_map:
                commands_to_run.extend(process_map[ext])

        if not commands_to_run:
            return

        # 3. Dispatch commands and handle recursion
        current_stream_provider = stream_provider_from_event
        for command_name in set(commands_to_run):  # Use set to avoid duplicate commands
            if command_name in COMMAND_MAP:
                command_class = COMMAND_MAP[command_name]
                processing_cmd = command_class(entity_id=entity_id, stream_provider=current_stream_provider)

                tqdm.write(f"Running {command_name} on entity {entity_id} at depth {depth}")

                # If we are processing a child entity (depth > 0), run its commands in a nested transaction.
                use_nested = depth > 0
                stream = target_world.dispatch_command(processing_cmd, use_nested_transaction=use_nested)

                sub_pbar: Optional[tqdm[Any]] = None

                async for event in stream:
                    if isinstance(event, NewEntityCreatedEvent):
                        tqdm.write(
                            f"  -> New entity {event.entity_id} ({event.filename or ''}) created from {entity_id}"
                        )
                        await _process_entity(
                            event.entity_id,
                            depth + 1,
                            pbar,
                            filename=event.filename,
                            stream_provider_from_event=event.stream_provider,
                        )
                    elif isinstance(event, ProgressStarted):
                        sub_pbar = tqdm(total=0, desc=f"  {command_name}", unit="B", unit_scale=True, leave=False)
                    elif isinstance(event, ProgressUpdate):
                        if sub_pbar:
                            if event.total is not None and sub_pbar.total != event.total:
                                sub_pbar.total = event.total
                            if event.current is not None:
                                sub_pbar.update(event.current - sub_pbar.n)
                            if event.message:
                                sub_pbar.set_description(f"  {command_name}: {event.message}")
                    elif isinstance(event, ProgressCompleted):
                        if sub_pbar:
                            # Make sure the bar is full
                            if sub_pbar.total and sub_pbar.n < sub_pbar.total:
                                sub_pbar.update(sub_pbar.total - sub_pbar.n)
                            sub_pbar.close()
                        if event.message:
                            tqdm.write(f"  -> Completed: {event.message}")
                    elif isinstance(event, ProgressError):
                        if sub_pbar:
                            sub_pbar.close()
                        tqdm.write(f"  -> Error processing {entity_id}: {event.message or str(event.exception)}")

                # The stream provider is single-use for non-seekable streams.
                # Set to None to force the next command to fetch a new one.
                current_stream_provider = None
            else:
                tqdm.write(f"Warning: Command '{command_name}' not found.")

    # Main execution starts here
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

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Registering assets", smoothing=0.0) as pbar:
        for file_path in files_to_process:
            tqdm.write(f"Process ({file_path})")
            pbar.set_postfix_str(file_path.name)
            try:
                mod_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime, tz=datetime.timezone.utc)
                pre_check_cmd = FindEntityByFilePropertiesCommand(
                    file_path=file_path.as_uri(), last_modified_at=mod_time
                )
                existing_entity_id = await target_world.dispatch_command(pre_check_cmd).get_one_value()

                if existing_entity_id:
                    skipped_count += 1
                    entity_id = existing_entity_id
                else:
                    register_cmd = RegisterLocalFileCommand(file_path=file_path)
                    entity_id = await target_world.dispatch_command(register_cmd).get_one_value()

                if entity_id:
                    success_count += 1
                    if process_map:
                        await _process_entity(entity_id, 0, pbar, filename=file_path.name)

            except Exception:
                error_count += 1
                console = Console()
                tqdm.write(f"Error processing file {file_path.name}")
                console.print(Traceback())
                if stop_on_error:
                    raise
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
    await target_world.dispatch_command(store_cmd).get_all_results()

    typer.secho("Asset storage process complete.", fg=typer.colors.GREEN)


@app.command(name="set-mime-type")
async def set_mime_type(
    entity_id: Annotated[int, typer.Argument(..., help="The ID of the entity.")],
    mime_type: Annotated[str, typer.Argument(..., help="The mime type to set.")],
):
    """
    Sets the mime type for an entity.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    typer.echo(f"Setting mime type for entity {entity_id} to {mime_type}...")

    set_cmd = SetMimeTypeCommand(entity_id=entity_id, mime_type=mime_type)
    await target_world.dispatch_command(set_cmd).get_all_results()

    typer.secho("Mime type set successfully.", fg=typer.colors.GREEN)


@app.command(name="show")
async def show_entity(
    entity_id: Annotated[int, typer.Argument(..., help="The ID of the entity to show.")],
):
    """
    Shows all components of a given entity in JSON format.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    async with target_world.transaction_manager() as tx:
        session = tx.session
        components_dict = await dam_ecs_functions.get_all_components_for_entity_as_dict(session, entity_id)
        if not components_dict:
            typer.secho(f"No components found for entity {entity_id}", fg=typer.colors.YELLOW)
            return
        from rich import print_json

        print_json(data=components_dict)


@app.command(name="auto-set-mime-type")
async def auto_set_mime_type(
    entity_id: Annotated[
        Optional[int],
        typer.Option(
            "--entity-id",
            "-e",
            help="The ID of the entity to process. If not provided, all entities will be processed.",
        ),
    ] = None,
):
    """
    Automatically sets the mime type for an entity or all entities.
    """
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    if entity_id:
        typer.echo(f"Automatically setting mime type for entity {entity_id}...")
    else:
        typer.echo("Automatically setting mime type for all entities...")

    if entity_id:
        set_cmd = AutoSetMimeTypeCommand(entity_id=entity_id)
        await target_world.dispatch_command(set_cmd).get_all_results()
    else:
        # TODO: Implement processing for all entities
        typer.secho("Processing all entities is not yet implemented.", fg=typer.colors.YELLOW)

    typer.secho("Mime type setting process complete.", fg=typer.colors.GREEN)
