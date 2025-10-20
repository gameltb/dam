"""Defines the CLI for managing assets in the DAM."""

from __future__ import annotations

import datetime
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any

import typer
from dam.commands.analysis_commands import AutoSetMimeTypeCommand
from dam.commands.asset_commands import (
    GetAssetFilenamesCommand,
    GetMimeTypeCommand,
)
from dam.core.executor import SystemExecutor
from dam.core.operations import AssetOperation
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.functions import ecs_functions as dam_ecs_functions
from dam.system_events.base import BaseSystemEvent, SystemResultEvent
from dam.system_events.entity_events import NewEntityCreatedEvent
from dam.system_events.progress import (
    ProgressCompleted,
    ProgressError,
    ProgressStarted,
    ProgressUpdate,
)
from dam.system_events.requests import PasswordRequest
from dam_fs.commands import (
    FindEntityByFilePropertiesCommand,
    RegisterLocalFileCommand,
    StoreAssetsCommand,
)
from rich import print_json
from rich.console import Console
from rich.table import Table
from rich.traceback import Traceback
from tqdm import tqdm

from dam_app.state import get_world
from dam_app.utils.async_typer import AsyncTyper

app = AsyncTyper()

MAX_RECURSION_DEPTH = 10


def _parse_process_options(process: list[str] | None, target_world: World) -> dict[str, list[str]]:
    """Parse the --process option strings into a map of keys to command names."""
    process_map: dict[str, list[str]] = {}
    if not process:
        return process_map

    for p in process:
        if ":" in p:
            try:
                key, command_name = p.split(":", 1)
                if key not in process_map:
                    process_map[key] = []
                process_map[key].append(command_name)
            except ValueError as e:
                raise ValueError(f"Invalid format for --process option: '{p}'. Must be 'key:CommandName'") from e
        else:
            operation_name = p
            operation = target_world.get_asset_operation(operation_name)

            if not operation:
                raise ValueError(f"Unknown operation '{operation_name}' specified.")

            supported_types = operation.get_supported_types()
            if not supported_types.get("mimetypes") and not supported_types.get("extensions"):
                typer.secho(
                    f"Operation '{operation_name}' does not support automatic type resolution.",
                    fg=typer.colors.YELLOW,
                )
                continue
            for mime_type in supported_types.get("mimetypes", []):
                if mime_type not in process_map:
                    process_map[mime_type] = []
                process_map[mime_type].append(operation_name)
            for extension in supported_types.get("extensions", []):
                if extension not in process_map:
                    process_map[extension] = []
                process_map[extension].append(operation_name)
    return process_map


async def _handle_progress_events(  # noqa: PLR0912
    stream: SystemExecutor[Any, BaseSystemEvent],
    operation_name: str,
    entity_id: int,
    depth: int,
    pbar: tqdm[Any],
    process_entity_callback: Callable[..., Coroutine[Any, Any, None]],
    target_world: World,
    process_map: dict[str, list[str]],
    passwords: list[str],
):
    """Handle progress events from a command stream."""
    sub_pbar: tqdm[Any] | None = None
    try:
        async for event in stream:
            if isinstance(event, NewEntityCreatedEvent):
                tqdm.write(f"  -> New entity {event.entity_id} ({event.filename or ''}) created from {entity_id}")
                await process_entity_callback(
                    target_world=target_world,
                    entity_id=event.entity_id,
                    depth=depth + 1,
                    pbar=pbar,
                    process_map=process_map,
                    filename=event.filename,
                    stream_provider_from_event=event.stream_provider,
                    passwords=passwords,
                )
            elif isinstance(event, PasswordRequest):
                tqdm.write("Password required for archive.")
                new_password = typer.prompt("Enter password", hide_input=True, default="").strip()
                if not new_password:
                    tqdm.write("No password provided, skipping archive.")
                    event.future.set_result(None)
                else:
                    if new_password not in passwords:
                        passwords.append(new_password)
                    event.future.set_result(new_password)

            elif isinstance(event, ProgressStarted):
                sub_pbar = tqdm(total=0, desc=f"  {operation_name}", unit="B", unit_scale=True, leave=False)
            elif isinstance(event, ProgressUpdate):
                if sub_pbar:
                    if event.total is not None and sub_pbar.total != event.total:
                        sub_pbar.total = event.total
                    if event.current is not None:
                        sub_pbar.update(event.current - sub_pbar.n)
                    if event.message:
                        sub_pbar.set_description(f"  {operation_name}: {event.message}")
            elif isinstance(event, ProgressCompleted):
                if sub_pbar:
                    if sub_pbar.total and sub_pbar.n < sub_pbar.total:
                        sub_pbar.update(sub_pbar.total - sub_pbar.n)
                    sub_pbar.close()
                    sub_pbar = None
                if event.message:
                    tqdm.write(f"  -> Completed: {event.message}")
            elif isinstance(event, ProgressError):
                if sub_pbar:
                    sub_pbar.close()
                    sub_pbar = None
                tqdm.write(f"  -> Error processing {entity_id}: {event.message or str(event.exception)}")
            elif isinstance(event, SystemResultEvent):
                break
    finally:
        if sub_pbar:
            sub_pbar.close()


def _collect_files(paths: list[Path], recursive: bool) -> list[Path]:
    """Collect all files to be processed from the given paths."""
    files_to_process: list[Path] = []
    for path in paths:
        if path.is_file():
            files_to_process.append(path)
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            files_to_process.extend(p for p in path.glob(pattern) if p.is_file())
    return files_to_process


async def _get_or_register_entity(target_world: World, file_path: Path) -> tuple[int | None, bool]:
    """Get an existing entity or register a new one. Returns (entity_id, was_skipped)."""
    mod_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime, tz=datetime.UTC)
    pre_check_cmd = FindEntityByFilePropertiesCommand(file_path=file_path.as_uri(), last_modified_at=mod_time)
    existing_entity_id = await target_world.dispatch_command(pre_check_cmd).get_one_value()

    if existing_entity_id:
        return existing_entity_id, True

    register_cmd = RegisterLocalFileCommand(file_path=file_path)
    entity_id = await target_world.dispatch_command(register_cmd).get_one_value()
    return entity_id, False


async def _get_action_for_operation(target_world: World, operation: AssetOperation, entity_id: int) -> str:
    """Determine the action to take for a given operation and entity."""
    if operation.check_command_class:
        check_cmd = operation.check_command_class(entity_id=entity_id)
        async with target_world.get_context(WorldTransaction)():
            already_exists = await target_world.dispatch_command(check_cmd).get_one_value()
        if already_exists:
            return "reprocess" if operation.reprocess_derived_command_class else "skip"
    return "add"


@app.command(name="add")
async def add_assets(
    paths: Annotated[
        list[Path],
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
        list[str] | None,
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
    """Register one or more local assets with the DAM."""
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    try:
        process_map = _parse_process_options(process, target_world)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    files_to_process = _collect_files(paths, recursive)
    if not files_to_process:
        typer.secho("No files found to process.", fg=typer.colors.YELLOW)
        return

    typer.echo(f"Found {len(files_to_process)} file(s) to process.")
    success_count = 0
    skipped_count = 0
    error_count = 0
    total_size = sum(p.stat().st_size for p in files_to_process)
    passwords: list[str] = []

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Registering assets", smoothing=0.0) as pbar:
        for file_path in files_to_process:
            tqdm.write(f"Process ({file_path})")
            pbar.set_postfix_str(file_path.name)
            try:
                entity_id, was_skipped = await _get_or_register_entity(target_world, file_path)
                if was_skipped:
                    skipped_count += 1
                else:
                    success_count += 1

                if entity_id and process_map:
                    await _process_entity(
                        target_world,
                        entity_id,
                        0,
                        pbar,
                        process_map,
                        filename=file_path.name,
                        stream_provider_from_event=None,
                        passwords=passwords,
                    )

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


async def _process_entity(  # noqa: PLR0912
    target_world: World,
    entity_id: int,
    depth: int,
    pbar: tqdm[Any],
    process_map: dict[str, list[str]],
    passwords: list[str],
    filename: str | None = None,
    stream_provider_from_event: Any | None = None,
):
    """Inner function to handle processing of a single entity."""
    if depth >= MAX_RECURSION_DEPTH:
        tqdm.write(f"Skipping entity {entity_id} at depth {depth}, limit reached.")
        return

    entity_filename: str | None = filename
    await target_world.dispatch_command(
        AutoSetMimeTypeCommand(entity_id=entity_id, stream_provider=stream_provider_from_event)
    ).get_all_results()
    mime_type_str = await target_world.dispatch_command(GetMimeTypeCommand(entity_id=entity_id)).get_one_value()

    if not entity_filename:
        filenames = await target_world.dispatch_command(GetAssetFilenamesCommand(entity_id=entity_id)).get_one_value()
        if filenames:
            entity_filename = filenames[0]

    pbar.set_postfix_str(f"Processing {entity_id} ({entity_filename or 'No Filename'})")

    commands_to_run: list[str] = []
    if mime_type_str and mime_type_str in process_map:
        commands_to_run.extend(process_map[mime_type_str])
    elif entity_filename:
        ext = Path(entity_filename).suffix.lower()
        if ext and ext in process_map:
            commands_to_run.extend(process_map[ext])

    if not commands_to_run:
        return

    for operation_name in set(commands_to_run):
        operation = target_world.get_asset_operation(operation_name)
        if not operation:
            tqdm.write(f"Warning: Operation '{operation_name}' not found.")
            continue

        action = await _get_action_for_operation(target_world, operation, entity_id)

        processing_cmd = None
        if action == "add" and operation.add_command_class:
            tqdm.write(f"Running {operation_name} on entity {entity_id} at depth {depth}")
            processing_cmd = operation.add_command_class(
                entity_id=entity_id,
                stream_provider=stream_provider_from_event,  # type: ignore [call-arg]
            )
        elif action == "reprocess" and operation.reprocess_derived_command_class:
            tqdm.write(
                f"Data for '{operation_name}' exists, reprocessing derived for entity {entity_id} at depth {depth}"
            )
            processing_cmd = operation.reprocess_derived_command_class(entity_id=entity_id)
        elif action == "skip":
            tqdm.write(f"Skipping operation '{operation_name}' for entity {entity_id}: data already exists.")
            continue

        if not processing_cmd:
            continue

        use_nested = depth > 0
        async with target_world.get_context(WorldTransaction)(use_nested_transaction=use_nested):
            stream = target_world.dispatch_command(processing_cmd)
            await _handle_progress_events(
                stream,
                operation_name,
                entity_id,
                depth,
                pbar,
                _process_entity,
                target_world,
                process_map,
                passwords,
            )


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
    """Copy registered local assets into the DAM's content-addressable storage."""
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    typer.echo(f"Storing assets for query: '{query}'...")

    store_cmd = StoreAssetsCommand(query=query)
    await target_world.dispatch_command(store_cmd).get_all_results()

    typer.secho("Asset storage process complete.", fg=typer.colors.GREEN)


@app.command(name="show")
async def show_entity(
    entity_id: Annotated[int, typer.Argument(..., help="The ID of the entity to show.")],
):
    """Show all components of a given entity in JSON format."""
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    async with target_world.get_context(WorldTransaction)() as tx:
        session = tx.session
        components_dict = await dam_ecs_functions.get_all_components_for_entity_as_dict(session, entity_id)
        if not components_dict:
            typer.secho(f"No components found for entity {entity_id}", fg=typer.colors.YELLOW)
            return
        print_json(data=components_dict)


@app.command(name="process")
async def process_entities(
    operation_name: Annotated[str, typer.Argument(..., help="The name of the operation to execute.")],
    entity_ids: Annotated[list[int], typer.Argument(..., help="The ID(s) of the entities to process.")],
):
    """Execute the 'add' action of a specific asset operation on one or more entities."""
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    operation = target_world.get_asset_operation(operation_name)
    if not operation:
        typer.secho(f"Unknown operation '{operation_name}' specified.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    command_class = operation.add_command_class
    typer.echo(f"Executing operation '{operation_name}' on {len(entity_ids)} entities...")

    for entity_id in entity_ids:
        typer.echo(f"Processing entity {entity_id}...")
        try:
            processing_cmd = command_class(entity_id=entity_id, stream_provider=None)  # type: ignore [call-arg]
            async with target_world.get_context(WorldTransaction)():
                stream = target_world.dispatch_command(processing_cmd)
                async for event in stream:
                    if isinstance(event, ProgressCompleted):
                        if event.message:
                            typer.echo(f"  -> Completed: {event.message}")
                    elif isinstance(event, ProgressError):
                        typer.secho(f"  -> Error: {event.message or str(event.exception)}", fg=typer.colors.RED)

        except Exception as e:
            typer.secho(f"An unexpected error occurred while processing entity {entity_id}: {e}", fg=typer.colors.RED)

    typer.secho("Processing complete.", fg=typer.colors.GREEN)


@app.command(name="remove-data")
async def remove_data(
    operation_name: Annotated[str, typer.Argument(..., help="The name of the operation to execute.")],
    entity_ids: Annotated[list[int], typer.Argument(..., help="The ID(s) of the entities to process.")],
):
    """Execute the 'remove' action of a specific asset operation on one or more entities."""
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    operation = target_world.get_asset_operation(operation_name)
    if not operation:
        typer.secho(f"Unknown operation '{operation_name}' specified.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not operation.remove_command_class:
        typer.secho(f"Operation '{operation_name}' does not support removing data.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    command_class = operation.remove_command_class
    typer.echo(f"Executing remove data for '{operation_name}' on {len(entity_ids)} entities...")

    for entity_id in entity_ids:
        typer.echo(f"Processing entity {entity_id}...")
        try:
            processing_cmd = command_class(entity_id=entity_id)
            async with target_world.get_context(WorldTransaction)():
                await target_world.dispatch_command(processing_cmd).get_all_results()
            typer.secho(f"  -> Data removed for entity {entity_id}.", fg=typer.colors.GREEN)

        except Exception as e:
            typer.secho(f"An unexpected error occurred while processing entity {entity_id}: {e}", fg=typer.colors.RED)

    typer.secho("Removal complete.", fg=typer.colors.GREEN)


@app.command(name="check-data")
async def check_data(
    operation_name: Annotated[str, typer.Argument(..., help="The name of the operation to execute.")],
    entity_ids: Annotated[list[int], typer.Argument(..., help="The ID(s) of the entities to check.")],
):
    """Execute the 'check' action of a specific asset operation on one or more entities."""
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    operation = target_world.get_asset_operation(operation_name)
    if not operation:
        typer.secho(f"Unknown operation '{operation_name}' specified.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not operation.check_command_class:
        typer.secho(f"Operation '{operation_name}' does not support checking data.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    command_class = operation.check_command_class
    typer.echo(f"Executing check data for '{operation_name}' on {len(entity_ids)} entities...")

    for entity_id in entity_ids:
        try:
            processing_cmd = command_class(entity_id=entity_id)
            async with target_world.get_context(WorldTransaction)():
                result = await target_world.dispatch_command(processing_cmd).get_one_value()

            if result:
                typer.secho(f"  -> Entity {entity_id}: Check PASSED (True)", fg=typer.colors.GREEN)
            else:
                typer.secho(f"  -> Entity {entity_id}: Check FAILED (False)", fg=typer.colors.RED)

        except Exception as e:
            typer.secho(f"An unexpected error occurred while checking entity {entity_id}: {e}", fg=typer.colors.RED)

    typer.secho("Check complete.", fg=typer.colors.GREEN)


@app.command(name="list-processes")
async def list_processes():
    """List all available asset processing operations."""
    target_world = get_world()
    if not target_world:
        raise typer.Exit(code=1)

    operations = target_world.get_all_asset_operations()

    if not operations:
        typer.secho("No asset processing operations found.", fg=typer.colors.YELLOW)
        return

    table = Table(title="Available Asset Processes", show_lines=True)
    table.add_column("Process Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")
    table.add_column("Supported MIME Types", style="green")
    table.add_column("Supported Extensions", style="blue")

    for op in sorted(operations, key=lambda x: x.name):
        supported_types = op.get_supported_types()
        mimetypes = ", ".join(supported_types.get("mimetypes", []))
        extensions = ", ".join(supported_types.get("extensions", []))
        table.add_row(op.name, op.description, mimetypes, extensions)

    console = Console()
    console.print(table)
