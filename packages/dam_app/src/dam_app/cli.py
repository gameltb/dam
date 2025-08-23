# --- Framework Imports for Systems ---
import asyncio
import json  # Added import for json.dumps
import traceback  # Import traceback for detailed error logging
import uuid  # For generating request_ids
from pathlib import Path
from typing import Any, List, Optional, Union
import logging

import typer  # Ensure typer is imported for annotations like typer.Context
from typing_extensions import Annotated

from dam.core import config as app_config
from dam.core.events import (
    AssetFileIngestionRequested,
    AssetReferenceIngestionRequested,
    FindEntityByHashQuery,
    FindSimilarImagesQuery,
    SemanticSearchQuery,  # For semantic search CLI
)
from dam.core.logging_config import setup_logging
from dam.core.stages import SystemStage
from dam.core.world import (
    World,
    clear_world_registry,
    create_and_register_all_worlds_from_settings,
    get_all_registered_worlds,
    get_world,
)
from dam.models.conceptual import CharacterConceptComponent
from dam.models.properties import FilePropertiesComponent
from dam.services import ecs_service as dam_ecs_service
from dam.systems import evaluation_systems
from dam.utils.async_typer import AsyncTyper
from dam.utils.media_utils import TranscodeError
from dam import DamPlugin

app = AsyncTyper(
    name="dam-cli",
    help="Digital Asset Management System CLI",
    add_completion=True,
    rich_markup_mode="raw",
)

class GlobalState:
    world_name: Optional[str] = None

global_state = GlobalState()

@app.command(name="list-worlds")
def cli_list_worlds():
    """Lists all configured and registered ECS worlds."""
    try:
        registered_worlds = get_all_registered_worlds()
        if not registered_worlds:
            typer.secho("No ECS worlds are currently registered or configured correctly.", fg=typer.colors.YELLOW)
            typer.echo("Check your DAM_WORLDS_CONFIG environment variable (JSON string or file path) and application logs for errors.")
            return

        typer.echo("Available ECS worlds:")
        for world_instance in registered_worlds:
            is_default = app_config.settings.DEFAULT_WORLD_NAME == world_instance.name
            default_marker = " (default)" if is_default else ""
            typer.echo(f"  - {world_instance.name}{default_marker}")

        configured_default = app_config.settings.DEFAULT_WORLD_NAME
        if configured_default:
            if not any(w.name == configured_default for w in registered_worlds):
                typer.secho(
                    f"\nWarning: The configured default world '{configured_default}' was not successfully registered. It might have configuration issues.",
                    fg=typer.colors.YELLOW,
                )
        elif registered_worlds:
            typer.secho(
                "\nNote: No specific default world is set. Operations might require explicit --world.",
                fg=typer.colors.YELLOW,
            )

    except Exception as e:
        typer.secho(f"Error listing worlds: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="add-asset")
async def cli_add_asset(
    ctx: typer.Context,
    path_str: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to the asset file or directory of asset files.",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    no_copy: Annotated[
        bool,
        typer.Option(
            "--no-copy",
            help="Add asset(s) by reference, without copying to DAM storage.",
        ),
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option("-r", "--recursive", help="Process directory recursively."),
    ] = False,
):
    if not global_state.world_name:
        typer.secho("Error: No world selected for add-asset. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(
            f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    input_path = Path(path_str)
    files_to_process: List[Path] = []

    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        typer.echo(f"Processing directory: {input_path} for world '{global_state.world_name}'")
        pattern = "*"
        if recursive:
            for item in input_path.rglob(pattern):
                if item.is_file():
                    files_to_process.append(item)
        else:
            for item in input_path.glob(pattern):
                if item.is_file():
                    files_to_process.append(item)
        if not files_to_process:
            typer.secho(f"No files found in {input_path}", fg=typer.colors.YELLOW)
            raise typer.Exit()
    else:
        typer.secho(f"Error: Path {input_path} is not a file or directory.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    total_files = len(files_to_process)
    typer.echo(f"Found {total_files} file(s) to process for world '{global_state.world_name}'.")

    processed_count = 0
    error_count = 0
    from dam.services import file_operations

    for filepath in files_to_process:
        processed_count += 1
        typer.echo(
            f"\nProcessing file {processed_count}/{total_files}: {filepath.name} (world: '{global_state.world_name}')"
        )
        try:
            original_filename, size_bytes = file_operations.get_file_properties(filepath)
        except Exception as e:
            typer.secho(f"  Error getting properties for {filepath}: {e}", fg=typer.colors.RED)
            error_count += 1
            continue

        event_to_dispatch: Optional[Any] = None
        if no_copy:
            event_to_dispatch = AssetReferenceIngestionRequested(
                filepath_on_disk=filepath,
                original_filename=original_filename,
                size_bytes=size_bytes,
                world_name=target_world.name,
            )
        else:
            event_to_dispatch = AssetFileIngestionRequested(
                filepath_on_disk=filepath,
                original_filename=original_filename,
                size_bytes=size_bytes,
                world_name=target_world.name,
            )

        try:
            async def dispatch_and_run_stages():
                if event_to_dispatch:
                    await target_world.dispatch_event(event_to_dispatch)
                    typer.secho(
                        f"  Dispatched {type(event_to_dispatch).__name__} for {original_filename}.",
                        fg=typer.colors.BLUE,
                    )
                    typer.echo(
                        f"  Running post-ingestion systems (e.g., metadata extraction) in world '{target_world.name}'..."
                    )
                    await target_world.execute_stage(SystemStage.METADATA_EXTRACTION)
                    typer.secho(f"  Post-ingestion systems completed for {original_filename}.", fg=typer.colors.GREEN)

            await dispatch_and_run_stages()

        except Exception as e:
            typer.secho(f"  Error processing file {filepath.name} via event system: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            error_count += 1

    typer.echo("\n--- Summary ---")
    typer.echo(f"World: '{target_world.name}'")
    typer.echo(f"Total files processed: {processed_count}")
    typer.echo(f"Errors encountered: {error_count}")
    if error_count > 0:
        typer.secho("Some files could not be processed. Check errors above.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="setup-db")
async def setup_db(ctx: typer.Context):
    if not global_state.world_name:
        typer.secho("Error: No world selected for setup-db. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(
            f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    typer.echo(f"Setting up database for world: '{target_world.name}'...")
    try:
        await target_world.create_db_and_tables()
        typer.secho(f"Database setup complete for world '{target_world.name}'.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during database setup for world '{target_world.name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)

@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    world: Annotated[
        Optional[str],
        typer.Option(
            "--world",
            "-w",
            help="Name of the ECS world to operate on. Uses default world if not specified.",
            envvar="DAM_CURRENT_WORLD",
        ),
    ] = None,
):
    setup_logging()
    initialized_worlds: List[World] = []
    try:
        initialized_worlds = create_and_register_all_worlds_from_settings(app_settings=app_config.settings)
    except Exception as e:
        typer.secho(f"Critical error: Could not initialize worlds from settings: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    for world_instance in initialized_worlds:
        world_instance.add_plugin(DamPlugin())

    try:
        from dam_psp import PspPlugin
        from dam_psp.systems import ingest_psp_isos_from_directory

        for world_instance in initialized_worlds:
            world_instance.add_plugin(PspPlugin())

        @app.command(name="ingest-psp-isos")
        async def cli_ingest_psp_isos(
            ctx: typer.Context,
            directory: Annotated[
                str, typer.Argument(..., help="Directory to scan for PSP ISOs and archives.", exists=True, resolve_path=True)
            ],
            passwords: Annotated[
                Optional[List[str]], typer.Option("--password", "-p", help="Password for encrypted archives.")
            ] = None,
        ):
            if not global_state.world_name:
                typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
                raise typer.Exit(code=1)

            target_world = get_world(global_state.world_name)
            if not target_world:
                typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
                raise typer.Exit(code=1)

            typer.echo(f"Starting PSP ISO ingestion for world '{target_world.name}' from directory: {directory}")

            async with target_world.db_session_maker() as session:
                try:
                    await ingest_psp_isos_from_directory(
                        session=session,
                        directory=directory,
                        passwords=passwords,
                    )
                    await session.commit()
                    typer.secho("Ingestion process completed.", fg=typer.colors.GREEN)
                except Exception as e:
                    await session.rollback()
                    typer.secho(f"An error occurred during ingestion: {e}", fg=typer.colors.RED)
                    typer.secho(traceback.format_exc(), fg=typer.colors.RED)
                    raise typer.Exit(code=1)

    except ImportError:
        logging.info("dam_psp plugin not installed. Skipping PSP ISO ingestion functionality.")

    try:
        from dam_semantic import SemanticPlugin
        from dam_semantic.systems import handle_semantic_search_query

        for world_instance in initialized_worlds:
            world_instance.add_plugin(SemanticPlugin())

        search_app = AsyncTyper(name="search", help="Search for assets using various methods.")
        app.add_typer(search_app)

        @search_app.command("semantic")
        async def cli_search_semantic(
            ctx: typer.Context,
            query: Annotated[str, typer.Option("--query", "-q", help="Text query for semantic search.")],
            top_n: Annotated[int, typer.Option("--top-n", "-n", help="Number of top results to return.")] = 10,
            model_name: Annotated[
                Optional[str], typer.Option("--model", "-m", help="Name of the sentence transformer model to use (optional).")
            ] = None,
        ):
            """Performs semantic search based on text query."""
            if not global_state.world_name:
                typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            target_world = get_world(global_state.world_name)
            if not target_world:
                typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
                raise typer.Exit(code=1)

            request_id = str(uuid.uuid4())
            query_event = SemanticSearchQuery(
                query_text=query,
                world_name=target_world.name,
                request_id=request_id,
                top_n=top_n,
                model_name=model_name,
            )

            typer.echo(
                f"Dispatching SemanticSearchQuery (Request ID: {request_id}) to world '{target_world.name}' for query: '{query[:100]}...'"
            )

            query_event.result_future = asyncio.get_running_loop().create_future()
            await target_world.dispatch_event(query_event)
            results = await query_event.result_future

            if not results:
                typer.secho(
                    f"No semantic matches found for query: '{query[:100]}...'. Request ID: {request_id}",
                    fg=typer.colors.YELLOW,
                )
                return

            typer.secho(f"--- Semantic Search Results (Request ID: {request_id}) ---", fg=typer.colors.GREEN)
            typer.echo(f"Found {len(results)} results for query '{query[:100]}...':")
            async with target_world.db_session_maker() as session:
                for entity, score, emb_comp in results:
                    fpc = await dam_ecs_service.get_component(session, entity.id, FilePropertiesComponent)
                    filename = fpc.original_filename if fpc else "N/A"
                    source_info = (
                        f"{emb_comp.source_component_name}.{emb_comp.source_field_name}" if emb_comp else "N/A"
                    )
                    typer.echo(
                        f"  - Entity ID: {entity.id}, Score: {score:.4f}, Filename: {filename}"
                        f"\n    Matched on: {source_info} (Model: {model_name})"
                    )

    except ImportError:
        logging.info("dam_semantic plugin not installed. Skipping semantic search functionality.")

    target_world_name_candidate: Optional[str] = None
    if world:
        target_world_name_candidate = world
    elif app_config.settings.DEFAULT_WORLD_NAME:
        target_world_name_candidate = app_config.settings.DEFAULT_WORLD_NAME

    global_state.world_name = target_world_name_candidate

    if global_state.world_name:
        selected_world_instance = get_world(global_state.world_name)
        if selected_world_instance:
            if ctx.invoked_subcommand:
                typer.echo(f"Operating on world: '{global_state.world_name}'")
        else:
            if ctx.invoked_subcommand not in ["list-worlds"]:
                typer.secho(
                    f"Error: World '{global_state.world_name}' is specified or default, but it's not correctly configured or registered.",
                    fg=typer.colors.RED,
                )
                registered_worlds_list = get_all_registered_worlds()
                if registered_worlds_list:
                    typer.echo("Available correctly configured worlds:")
                    for w_instance in registered_worlds_list:
                        typer.echo(f"  - {w_instance.name}")
                else:
                    typer.echo("No worlds appear to be correctly configured.")
                raise typer.Exit(code=1)
    elif ctx.invoked_subcommand not in ["list-worlds"]:
        typer.secho(
            "Error: No ECS world specified and no default world could be determined. "
            "Use --world <world_name>, set DAM_DEFAULT_WORLD_NAME, or configure a 'default' world in DAM_WORLDS_CONFIG.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if ctx.invoked_subcommand is None:
        current_selection_info = (
            f"Current default/selected world: '{global_state.world_name}' (Use --world to change)"
            if global_state.world_name
            else "No default world selected. Use --world <world_name> or see 'list-worlds'."
        )
        typer.echo(current_selection_info)
        if not get_all_registered_worlds() and not app_config.settings.worlds:
            typer.secho("No DAM worlds seem to be configured. Please set DAM_WORLDS_CONFIG.", fg=typer.colors.YELLOW)

def run_cli_directly():
    clear_world_registry()
    app()

if __name__ == "__main__":
    run_cli_directly()
