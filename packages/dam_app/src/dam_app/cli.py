# --- Framework Imports for Systems ---
import io
import logging
import traceback  # Import traceback for detailed error logging
import uuid  # For generating request_ids
from pathlib import Path
from typing import List, Optional

import typer  # Ensure typer is imported for annotations like typer.Context
from dam import DamPlugin
from dam.core import config as app_config
from dam.core.logging_config import setup_logging
from dam.core.world import (
    World,
    clear_world_registry,
    create_and_register_all_worlds_from_settings,
    get_all_registered_worlds,
    get_world,
)
from dam.functions import ecs_functions as dam_ecs_functions
from dam.utils.async_typer import AsyncTyper
from dam_archive.commands import IngestAssetsCommand
from dam_fs.events import AssetsReadyForMetadataExtraction
from dam_fs.models import FilePropertiesComponent
from typing_extensions import Annotated

from .commands import IngestAssetStreamCommand

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
            typer.echo(
                "Check your DAM_WORLDS_CONFIG environment variable (JSON string or file path) and application logs for errors."
            )
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
        return


async def add_asset(world: World, files_to_process: List[Path]):
    total_files = len(files_to_process)
    typer.echo(f"Found {total_files} file(s) to process for world '{world.name}'.")

    processed_count = 0
    error_count = 0

    async with world.db_session_maker() as session:
        for filepath in files_to_process:
            processed_count += 1
            typer.echo(f"\nProcessing file {processed_count}/{total_files}: {filepath.name}")
            try:
                # Create a new entity for this asset first
                entity = await dam_ecs_functions.create_entity(session)
                await session.flush()  # Ensure entity gets an ID

                # Read file content into an in-memory stream
                with open(filepath, "rb") as f:
                    file_content_stream = io.BytesIO(f.read())

                # Create and dispatch the initial command
                command = IngestAssetStreamCommand(
                    entity=entity,
                    file_content=file_content_stream,
                    original_filename=filepath.name,
                    world_name=world.name,
                )

                await world.dispatch_command(command)

                # The command handler now handles everything. We just need to commit the session.
                await session.commit()
                typer.secho(
                    f"  Successfully dispatched ingestion command for '{filepath.name}'.", fg=typer.colors.GREEN
                )

            except Exception as e:
                await session.rollback()
                typer.secho(f"  Error processing file {filepath.name}: {e}", fg=typer.colors.RED)
                typer.secho(traceback.format_exc(), fg=typer.colors.RED)
                error_count += 1
                # Continue to next file

    typer.echo("\n--- Summary ---")
    typer.echo(f"World: '{world.name}'")
    typer.echo(f"Total files attempted: {processed_count}")
    typer.echo(f"Errors encountered: {error_count}")
    if error_count > 0:
        typer.secho("Some files could not be processed. Check errors above.", fg=typer.colors.RED)
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
    recursive: Annotated[
        bool,
        typer.Option("-r", "--recursive", help="Process directory recursively."),
    ] = False,
):
    """
    Adds one or more assets to the DAM, initiating the new event-driven ingestion pipeline.
    """
    if not global_state.world_name:
        typer.secho("Error: No world selected for add-asset. Use --world <world_name>.", fg=typer.colors.RED)
        return

    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(
            f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED
        )
        return

    input_path = Path(path_str)
    files_to_process: List[Path] = []

    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        typer.echo(f"Scanning directory: {input_path} for world '{global_state.world_name}'")
        pattern = "**/*" if recursive else "*"
        files_to_process.extend(p for p in input_path.glob(pattern) if p.is_file())
        if not files_to_process:
            typer.secho(f"No files found in {input_path}", fg=typer.colors.YELLOW)
            return
    else:
        typer.secho(f"Error: Path {input_path} is not a file or directory.", fg=typer.colors.RED)
        return

    await add_asset(target_world, files_to_process)


@app.command(name="ingest")
async def cli_ingest(
    ctx: typer.Context,
    paths: Annotated[
        List[str],
        typer.Argument(
            ...,
            help="Paths to the asset files or directories.",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    recursive: Annotated[
        bool,
        typer.Option("-r", "--recursive", help="Process directories recursively."),
    ] = False,
    passwords: Annotated[
        Optional[List[str]], typer.Option("--password", "-p", help="Password for encrypted archives.")
    ] = None,
):
    """
    Ingests assets from files or directories, expanding archives.
    """
    if not global_state.world_name:
        typer.secho("Error: No world selected for ingest. Use --world <world_name>.", fg=typer.colors.RED)
        return

    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(
            f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED
        )
        return

    files_to_process: List[str] = []
    for path_str in paths:
        input_path = Path(path_str)
        if input_path.is_file():
            files_to_process.append(str(input_path))
        elif input_path.is_dir():
            pattern = "**/*" if recursive else "*"
            files_to_process.extend(str(p) for p in input_path.glob(pattern) if p.is_file())

    if not files_to_process:
        typer.secho("No files found to ingest.", fg=typer.colors.YELLOW)
        return

    command = IngestAssetsCommand(file_paths=files_to_process, passwords=passwords)

    try:
        new_entity_ids = await target_world.dispatch_command(command)
        typer.secho(f"Successfully ingested {len(new_entity_ids)} assets.", fg=typer.colors.GREEN)

        if new_entity_ids:
            typer.echo("Dispatching assets for metadata extraction...")
            await target_world.send_event(AssetsReadyForMetadataExtraction(entity_ids=new_entity_ids))
            typer.secho("Metadata extraction event dispatched.", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"An error occurred during ingestion: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
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

    from dam_app.plugin import AppPlugin

    for world_instance in initialized_worlds:
        world_instance.add_plugin(AppPlugin())

    try:
        from dam_fs import FsPlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(FsPlugin())
    except ImportError:
        logging.info("dam_fs plugin not installed. Skipping fs functionality.")

    try:
        from dam_source import SourcePlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(SourcePlugin())
    except ImportError:
        logging.info("dam_source plugin not installed. Skipping source functionality.")

    try:
        from dam_sire import SirePlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(SirePlugin())
    except ImportError:
        logging.info("dam_sire plugin not installed. Skipping sire functionality.")

    try:
        from dam_media_image import ImagePlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(ImagePlugin())
    except ImportError:
        logging.info("dam_media_image plugin not installed. Skipping image functionality.")

    try:
        from dam_media_audio import AudioPlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(AudioPlugin())
    except ImportError:
        logging.info("dam_media_audio plugin not installed. Skipping audio functionality.")

    try:
        from dam_media_transcode import TranscodePlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(TranscodePlugin())
    except ImportError:
        logging.info("dam_media_transcode plugin not installed. Skipping transcode functionality.")

    try:
        from dam_archive import ArchivePlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(ArchivePlugin())
    except ImportError:
        logging.info("dam_archive plugin not installed. Skipping archive functionality.")

    try:
        from dam_psp import PspPlugin

        for world_instance in initialized_worlds:
            world_instance.add_plugin(PspPlugin())
    except ImportError:
        logging.info("dam_psp plugin not installed. Skipping PSP ISO ingestion functionality.")

    try:
        from dam_semantic import SemanticPlugin
        from dam_semantic.commands import SemanticSearchCommand  # For semantic search CLI

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
                Optional[str],
                typer.Option("--model", "-m", help="Name of the sentence transformer model to use (optional)."),
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
            query_command = SemanticSearchCommand(
                query_text=query,
                request_id=request_id,
                top_n=top_n,
                model_name=model_name,
            )

            typer.echo(
                f"Dispatching SemanticSearchCommand (Request ID: {request_id}) to world '{target_world.name}' for query: '{query[:100]}...'"
            )

            command_result = await target_world.dispatch_command(query_command)
            results = command_result.results[0] if command_result.results else None

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
                    fpc = await dam_ecs_functions.get_component(session, entity.id, FilePropertiesComponent)
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
