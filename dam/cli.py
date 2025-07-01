# --- Framework Imports for Systems ---
import asyncio
import json  # Added import for json.dumps
import traceback  # Import traceback for detailed error logging
import uuid  # For generating request_ids
from pathlib import Path
from typing import Any, List, Optional, Union

from typing_extensions import Annotated
import typer # Ensure typer is imported for annotations like typer.Context

from dam.core import config as app_config
from dam.utils.async_typer import AsyncTyper
from dam.core.events import (
    AssetFileIngestionRequested,
    AssetReferenceIngestionRequested,
    FindEntityByHashQuery,
    FindSimilarImagesQuery,
)
from dam.core.logging_config import setup_logging

# ResourceManager and WorldScheduler will be part of World instances
# from dam.core.resources import FileOperationsResource, ResourceManager # Removed
# from dam.core.systems import WorldContext, WorldScheduler # Removed
from dam.core.stages import SystemStage  # Keep for system execution if needed directly by CLI
from dam.core.world import (  # Added
    World,
    clear_world_registry,
    create_and_register_all_worlds_from_settings,
    get_all_registered_worlds,
    get_world,
)
from dam.services import (
    file_operations,
    transcode_service,
    world_service,
    character_service, # Added character_service
    ecs_service as dam_ecs_service, # Alias to avoid conflict with local ecs_service module
    semantic_service, # Added semantic_service
)
from dam.models.conceptual import CharacterConceptComponent # For displaying character info
from dam.models.properties import FilePropertiesComponent # For displaying asset info
from dam.core.events import SemanticSearchQuery # For semantic search CLI

from dam.systems import evaluation_systems
from dam.utils.media_utils import TranscodeError

# Systems will be imported and then registered manually to worlds.
# from dam import systems as dam_all_systems # Import the systems package
# No global ResourceManager or WorldScheduler here anymore.
# They are instantiated per World.
# Import specific system functions that need to be registered
# This is an example; a more dynamic way might be needed for many systems.

app = AsyncTyper(
    name="dam-cli",
    help="Digital Asset Management System CLI",
    add_completion=True,
    rich_markup_mode="raw",  # Attempt to mitigate CliRunner issues with Rich
)


# Global state for Typer context, primarily for holding the selected world name
class GlobalState:
    world_name: Optional[str] = None


global_state = GlobalState()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    world: Annotated[
        Optional[str],
        typer.Option(
            "--world",
            "-w",
            help="Name of the ECS world to operate on. Uses default world if not specified.",
            envvar="DAM_CURRENT_WORLD",  # Allow setting via environment variable
        ),
    ] = None,
):
    """
    Main CLI callback to set up logging and handle global options like --world.
    """
    setup_logging()
    # Initialize all worlds from settings when CLI starts
    # This also populates the world registry.
    initialized_worlds: List[World] = []
    try:
        # Pass the global app_config.settings to ensure CLI uses the correct one
        initialized_worlds = create_and_register_all_worlds_from_settings(app_settings=app_config.settings)
        # typer.echo(f"Initialized {len(initialized_worlds)} worlds.") # Optional debug
    except Exception as e:
        # If basic world loading from settings fails, it's a critical startup error.
        typer.secho(f"Critical error: Could not initialize worlds from settings: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # After worlds are created, register systems to each world instance
    # This is a simplified example; a more robust mechanism might involve iterating
    # through system modules or using entry points.
    # Task 2.2: Use the centralized world_registrar (now in world_setup)
    from dam.core.world_setup import register_core_systems

    for world_instance in initialized_worlds:
        register_core_systems(world_instance)

    # Determine the target world name
    # Priority: CLI option > Environment Variable > app_config.settings.DEFAULT_WORLD_NAME
    target_world_name_candidate: Optional[str] = None
    if world:
        target_world_name_candidate = world
    elif app_config.settings.DEFAULT_WORLD_NAME:
        target_world_name_candidate = app_config.settings.DEFAULT_WORLD_NAME

    global_state.world_name = target_world_name_candidate

    # Validate the selected world and ensure it's registered, unless it's a command that doesn't need a world
    if global_state.world_name:
        selected_world_instance = get_world(global_state.world_name)
        if selected_world_instance:
            if ctx.invoked_subcommand:  # Only print if a subcommand is being invoked
                typer.echo(f"Operating on world: '{global_state.world_name}'")
        else:
            # This means the name was determined, but no such world is registered (e.g., config error for that specific world)
            if ctx.invoked_subcommand not in ["list-worlds"]:  # Allow list-worlds
                typer.secho(
                    f"Error: World '{global_state.world_name}' is specified or default, but it's not correctly configured or registered.",
                    fg=typer.colors.RED,
                )
                # List available valid worlds to help user
                registered_worlds_list = get_all_registered_worlds()
                if registered_worlds_list:
                    typer.echo("Available correctly configured worlds:")
                    for w_instance in registered_worlds_list:
                        typer.echo(f"  - {w_instance.name}")
                else:
                    typer.echo("No worlds appear to be correctly configured.")
                raise typer.Exit(code=1)
    elif ctx.invoked_subcommand not in ["list-worlds"]:
        # No world name could be determined (no --world, no env var, no default in settings)
        # And the command is not 'list-worlds'
        typer.secho(
            "Error: No ECS world specified and no default world could be determined. "
            "Use --world <world_name>, set DAM_DEFAULT_WORLD_NAME, or configure a 'default' world in DAM_WORLDS_CONFIG.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if ctx.invoked_subcommand is None:
        # Typer shows help by default.
        # If a default world is identifiable, we could print it.
        current_selection_info = (
            f"Current default/selected world: '{global_state.world_name}' (Use --world to change)"
            if global_state.world_name
            else "No default world selected. Use --world <world_name> or see 'list-worlds'."
        )

        typer.echo(current_selection_info)

        if (
            not get_all_registered_worlds() and not app_config.settings.worlds
        ):  # Check both raw config and successfully registered
            typer.secho("No DAM worlds seem to be configured. Please set DAM_WORLDS_CONFIG.", fg=typer.colors.YELLOW)


@app.command(name="list-worlds")
def cli_list_worlds():
    """Lists all configured and registered ECS worlds."""
    try:
        # Worlds should have been registered by the main_callback's call to create_and_register_all_worlds_from_settings()
        registered_worlds = get_all_registered_worlds()

        if not registered_worlds:
            typer.secho("No ECS worlds are currently registered or configured correctly.", fg=typer.colors.YELLOW)
            typer.echo(
                "Check your DAM_WORLDS_CONFIG environment variable (JSON string or file path) and application logs for errors."
            )
            # The function will naturally proceed to print "Available ECS worlds:" which will be an empty list,
            # and then handle default world warnings if applicable, or exit cleanly.
            # This is better than an early return making subsequent code unreachable.

        typer.echo("Available ECS worlds:")  # Will print this, then loop (or not if empty)
        for world_instance in registered_worlds:
            is_default = app_config.settings.DEFAULT_WORLD_NAME == world_instance.name
            default_marker = " (default)" if is_default else ""
            # You can add more details here, e.g., world_instance.config.DATABASE_URL
            typer.echo(f"  - {world_instance.name}{default_marker}")

        # The logic for DEFAULT_WORLD_NAME validation is now implicitly handled by Settings model and get_world
        # but we can still provide a note if the explicitly set default isn't among the *successfully registered* ones.
        configured_default = app_config.settings.DEFAULT_WORLD_NAME
        if configured_default:
            if not any(w.name == configured_default for w in registered_worlds):
                typer.secho(
                    f"\nWarning: The configured default world '{configured_default}' was not successfully registered. "
                    "It might have configuration issues.",
                    fg=typer.colors.YELLOW,
                )
        elif registered_worlds:  # Worlds exist, but no default was determined by settings
            typer.secho(
                "\nNote: No specific default world is set. Operations might require explicit --world.",
                fg=typer.colors.YELLOW,
            )
        # If registered_worlds is empty AND configured_default is None, nothing specific is printed here, which is fine.

    except Exception as e:  # Catch any unexpected errors during listing
        typer.secho(f"Error listing worlds: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="add-asset")
async def cli_add_asset(  # Made async
    ctx: typer.Context,  # Added context to access global_state if needed, though --world handles it
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
    """
    Adds new asset file(s) to the specified ECS world.
    """
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
            for item in input_path.glob(pattern):  # Use glob for consistency, iterdir is also fine
                if item.is_file():
                    files_to_process.append(item)
        if not files_to_process:
            typer.secho(f"No files found in {input_path}", fg=typer.colors.YELLOW)
            raise typer.Exit()
    else:  # Should be caught by exists=True # pragma: no cover
        typer.secho(f"Error: Path {input_path} is not a file or directory.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    total_files = len(files_to_process)
    typer.echo(f"Found {total_files} file(s) to process for world '{global_state.world_name}'.")

    processed_count = 0
    error_count = 0
    # added_count, linked_count removed as they are not directly determined here anymore

    # FileStorageService is no longer directly obtained here, as systems will get it via DI.

    for filepath in files_to_process:
        processed_count += 1
        typer.echo(
            f"\nProcessing file {processed_count}/{total_files}: {filepath.name} (world: '{global_state.world_name}')"
        )
        try:
            original_filename, size_bytes, mime_type = file_operations.get_file_properties(filepath)
        except Exception as e:  # pragma: no cover
            typer.secho(f"  Error getting properties for {filepath}: {e}", fg=typer.colors.RED)
            error_count += 1
            continue

        event_to_dispatch: Optional[Any] = None
        if no_copy:
            event_to_dispatch = AssetReferenceIngestionRequested(
                filepath_on_disk=filepath,
                original_filename=original_filename,
                mime_type=mime_type,
                size_bytes=size_bytes,
                world_name=target_world.name,
            )
        else:
            event_to_dispatch = AssetFileIngestionRequested(
                filepath_on_disk=filepath,
                original_filename=original_filename,
                mime_type=mime_type,
                size_bytes=size_bytes,
                world_name=target_world.name,
            )

        try:

            async def dispatch_and_run_stages():
                if event_to_dispatch:
                    await target_world.dispatch_event(event_to_dispatch)
                    # Assuming event processing is successful, we can log.
                    # The actual entity ID and created_new status are not directly available here.
                    # We rely on logs from the system or subsequent queries for confirmation.
                    typer.secho(
                        f"  Dispatched {type(event_to_dispatch).__name__} for {original_filename}.",
                        fg=typer.colors.BLUE,
                    )
                    # If an entity was processed by the event, METADATA_EXTRACTION stage will run
                    # on entities marked by NeedsMetadataExtractionComponent.
                    # This marker is added by the ingestion event handlers.
                    typer.echo(
                        f"  Running post-ingestion systems (e.g., metadata extraction) in world '{target_world.name}'..."
                    )
                    await target_world.execute_stage(SystemStage.METADATA_EXTRACTION)
                    typer.secho(f"  Post-ingestion systems completed for {original_filename}.", fg=typer.colors.GREEN)

            await dispatch_and_run_stages()  # Await the async inner function
            # Note: We can't easily get added_count/linked_count here without querying.
            # For simplicity, we'll just count processed files.
            # If specific feedback is needed, systems would need to update a result resource.

        except Exception as e:
            # Event dispatch or stage execution itself failed.
            typer.secho(f"  Error processing file {filepath.name} via event system: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            error_count += 1
        # finally: # db_session is not defined in this scope anymore; session management is internal to world methods.
        # if db_session:
        # db_session.close()

    typer.echo("\n--- Summary ---")
    typer.echo(f"World: '{target_world.name}'")
    typer.echo(f"Total files processed: {processed_count}")
    # added_count and linked_count are no longer directly available from event dispatch
    # typer.echo(f"New assets added: {added_count}")
    # typer.echo(f"Existing assets linked: {linked_count}")
    typer.echo(f"Errors encountered: {error_count}")
    if error_count > 0:
        typer.secho("Some files could not be processed. Check errors above.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="setup-db")
async def setup_db(ctx: typer.Context):  # Added context, made async
    """
    Initializes the database and creates tables for the specified/default ECS world.
    """
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
        await target_world.create_db_and_tables()  # World method calls its db_manager
        typer.secho(f"Database setup complete for world '{target_world.name}'.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during database setup for world '{target_world.name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="find-file-by-hash")
async def cli_find_file_by_hash(  # Made async
    ctx: typer.Context,  # Added context
    hash_value_arg: Annotated[
        str, typer.Argument(..., help="The hash value of the file to search for.", metavar="HASH_VALUE")
    ],
    hash_type: Annotated[str, typer.Option(help="Type of hash ('sha256', 'md5'). Default 'sha256'.")] = "sha256",
    target_filepath: Annotated[
        Optional[Path],
        typer.Option("--file", "-f", help="File path to calculate hash from.", exists=True, resolve_path=True),
    ] = None,
):
    """
    Finds an asset by content hash in the specified ECS world.
    """
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(
            f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    actual_hash_value = hash_value_arg
    actual_hash_type = hash_type.lower()

    if target_filepath:
        typer.echo(
            f"Calculating {actual_hash_type} hash for file: {target_filepath} (for world '{target_world.name}')..."
        )
        try:
            if actual_hash_type == "sha256":
                actual_hash_value = file_operations.calculate_sha256(target_filepath)
            elif actual_hash_type == "md5":
                actual_hash_value = file_operations.calculate_md5(target_filepath)
            else:
                typer.secho(f"Unsupported hash type for calculation: {actual_hash_type}", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            typer.echo(f"Calculated {actual_hash_type} hash: {actual_hash_value}")
        except Exception as e:
            typer.secho(f"Error calculating hash for {target_filepath}: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    request_id = str(uuid.uuid4())
    query_event = FindEntityByHashQuery(
        hash_value=actual_hash_value,
        hash_type=actual_hash_type,
        world_name=target_world.name,
        request_id=request_id,
    )
    # Create a future for the result - this needs to be done inside the async function
    # that will run the event loop.

    typer.echo(
        f"Dispatching FindEntityByHashQuery (Request ID: {request_id}) to world '{target_world.name}' for hash: {actual_hash_value}"
    )

    async def dispatch_query_and_get_result():
        # Create future inside the async function where the loop is running
        query_event.result_future = asyncio.get_running_loop().create_future()
        await target_world.dispatch_event(query_event)  # Event handler will set the future's result
        try:
            # Wait for the result from the future with a timeout
            details = await asyncio.wait_for(query_event.result_future, timeout=10.0)
            if details:
                typer.secho(f"--- Asset Found (Request ID: {request_id}) ---", fg=typer.colors.GREEN)
                typer.echo(f"Entity ID: {details.get('entity_id')}")

                if "FilePropertiesComponent" in details.get("components", {}):
                    fpc = details["components"]["FilePropertiesComponent"]
                    typer.echo(f"  Original Filename: {fpc.get('original_filename')}")
                    typer.echo(f"  Size: {fpc.get('file_size_bytes')} bytes")
                    typer.echo(f"  MIME Type: {fpc.get('mime_type')}")

                if "ContentHashSHA256Component" in details.get("components", {}):
                    sha256c = details["components"]["ContentHashSHA256Component"]
                    typer.echo(f"  SHA256: {sha256c.get('hash_value')}")

                if "ContentHashMD5Component" in details.get("components", {}):
                    md5c = details["components"]["ContentHashMD5Component"]
                    typer.echo(f"  MD5: {md5c.get('hash_value')}")

                if "FileLocationComponent" in details.get("components", {}):
                    flcs = details["components"]["FileLocationComponent"]
                    typer.echo("  File Locations:")
                    for flc in flcs:
                        typer.echo(f"    - Contextual Filename: {flc.get('contextual_filename')}")
                        typer.echo(f"      Storage Type: {flc.get('storage_type')}")
                        typer.echo(f"      Path/Key: {flc.get('physical_path_or_key')}")
                        typer.echo(f"      Content ID: {flc.get('content_identifier')}")
                # Add more component details as needed
            else:
                typer.secho(
                    f"No asset found for hash {actual_hash_value} (Type: {actual_hash_type}). Request ID: {request_id}",
                    fg=typer.colors.YELLOW,
                )
        except asyncio.TimeoutError:
            typer.secho(f"Query timed out for Request ID: {request_id}.", fg=typer.colors.RED)
        except Exception as e:
            # This catches exceptions set on the future by the handler, or other await errors
            typer.secho(
                f"Query failed for Request ID: {request_id}. Error: {e}",
                fg=typer.colors.RED,
            )
            # Optionally print traceback if the exception from future doesn't have enough context
            # if not isinstance(e, (YourCustomQueryError)): # Check if it's an error from the handler
            #     typer.secho(traceback.format_exc(), fg=typer.colors.RED)

    try:
        await dispatch_query_and_get_result()  # Await async call
    except Exception as e:  # Catch errors from dispatch_event itself if any occur before future handling
        typer.secho(f"Error during query dispatch setup for world '{target_world.name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="find-similar-images")
async def cli_find_similar_images(  # Made async
    ctx: typer.Context,  # Added context
    image_filepath_str: Annotated[
        str, typer.Argument(..., help="Path to image for similarity search.", resolve_path=True, exists=True)
    ],
    phash_threshold: Annotated[int, typer.Option(help="pHash threshold.")] = 4,
    ahash_threshold: Annotated[int, typer.Option(help="aHash threshold.")] = 4,
    dhash_threshold: Annotated[int, typer.Option(help="dHash threshold.")] = 4,
):
    """
    Finds similar images in the specified ECS world.
    """
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(
            f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    image_filepath = Path(image_filepath_str)
    request_id = str(uuid.uuid4())

    query_event = FindSimilarImagesQuery(
        image_path=image_filepath,
        phash_threshold=phash_threshold,
        ahash_threshold=ahash_threshold,
        dhash_threshold=dhash_threshold,
        world_name=target_world.name,
        request_id=request_id,
    )

    typer.echo(
        f"Dispatching FindSimilarImagesQuery (Request ID: {request_id}) to world '{target_world.name}' for image: {image_filepath.name}"
    )

    async def dispatch_query_and_get_result():
        query_event.result_future = asyncio.get_running_loop().create_future()
        await target_world.dispatch_event(query_event)
        try:
            results = await asyncio.wait_for(
                query_event.result_future, timeout=30.0
            )  # Increased timeout for similarity search

            if results:
                typer.secho(f"--- Similar Images Found (Request ID: {request_id}) ---", fg=typer.colors.GREEN)
                # Check if the result is an error indicator (list with a single dict containing 'error')
                if len(results) == 1 and "error" in results[0]:
                    error_message = results[0].get("error", "Error in processing similarity query.")
                    typer.secho(f"Info: {error_message}", fg=typer.colors.YELLOW)
                else:
                    typer.echo(f"Found {len(results)} similar image(s) to '{image_filepath.name}':")
                    for match in results:
                        typer.echo(
                            f"  - Entity ID: {match.get('entity_id')}, "
                            f"Filename: {match.get('original_filename', 'N/A')}, "
                            f"Distance: {match.get('distance')} ({match.get('hash_type')})"
                        )
            else:  # Results is None or an empty list
                typer.secho(
                    f"No similar images found for '{image_filepath.name}'. Request ID: {request_id}",
                    fg=typer.colors.YELLOW,
                )
        except asyncio.TimeoutError:
            typer.secho(f"Similarity query timed out for Request ID: {request_id}.", fg=typer.colors.RED)
        except Exception as e:
            typer.secho(
                f"Similarity query failed for Request ID: {request_id}. Error: {e}",
                fg=typer.colors.RED,
            )
            # typer.secho(traceback.format_exc(), fg=typer.colors.RED) # Optionally show full traceback

    try:
        await dispatch_query_and_get_result()  # Await async call
    except Exception as e:  # Catch errors from dispatch_event itself
        typer.secho(f"Error during similarity query dispatch to world '{target_world.name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="export-world")
def cli_export_world(
    ctx: typer.Context,  # Added context
    filepath_str: Annotated[str, typer.Argument(..., help="Path to export JSON file to.", resolve_path=True)],
):
    """Exports the specified ECS world to a JSON file."""
    if not global_state.world_name:
        typer.secho("Error: No world selected for export. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    export_path = Path(filepath_str)
    if export_path.is_dir():
        typer.secho(f"Error: Export path {export_path} is a directory.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if export_path.exists() and not typer.confirm(f"File {export_path} exists. Overwrite?", default=False):
        typer.echo("Export cancelled.")
        raise typer.Exit()

    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(
            f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    typer.echo(f"Exporting ECS world '{target_world.name}' to: {export_path}")
    try:
        # Service function now takes World object and handles its own session
        world_service.export_ecs_world_to_json(target_world, export_path)
        typer.secho(f"ECS world '{target_world.name}' exported to {export_path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error exporting world '{target_world.name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="import-world")
def cli_import_world(
    ctx: typer.Context,  # Added context
    filepath_str: Annotated[
        str, typer.Argument(..., help="Path to ECS world JSON file to import.", resolve_path=True, exists=True)
    ],
    merge: Annotated[bool, typer.Option("--merge", help="Merge with existing data.")] = False,
):
    """Imports an ECS world from a JSON file into the specified ECS world."""
    if not global_state.world_name:
        typer.secho("Error: No world selected for import. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(
            f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    import_path = Path(filepath_str)
    typer.echo(f"Importing ECS world from: {import_path} into world '{target_world.name}'")
    if merge:
        typer.echo("Merge mode enabled.")

    try:
        # Service function now takes World object and handles its own session
        world_service.import_ecs_world_from_json(target_world, import_path, merge=merge)
        typer.secho(f"ECS world imported into '{target_world.name}' from {import_path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error importing into world '{target_world.name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


# Remove placeholder command if no longer needed
# @app.command(name="query-assets-placeholder", hidden=True) ...


@app.command(name="merge-worlds-db")
def cli_merge_worlds_db(
    source_world: Annotated[str, typer.Argument(help="Name of the source ECS world.")],
    target_world: Annotated[str, typer.Argument(help="Name of the target ECS world.")],
    # Add strategy option later if more are implemented
):
    """
    Merges entities from a source ECS world into a target ECS world (DB-to-DB).
    Currently uses 'add_new' strategy: all source entities are added as new to the target.
    """
    typer.echo(f"Merging world '{source_world}' into world '{target_world}' using 'add_new' strategy...")

    source_world_instance = get_world(source_world)
    target_world_instance = get_world(target_world)

    if not source_world_instance:
        typer.secho(f"Error: Source world '{source_world}' not found or not initialized.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not target_world_instance:
        typer.secho(f"Error: Target world '{target_world}' not found or not initialized.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if source_world_instance.name == target_world_instance.name:  # Compare by actual name from instance
        typer.secho("Error: Source and target worlds cannot be the same for merge operation.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Service function now takes World objects and handles its own sessions
        world_service.merge_ecs_worlds_db_to_db(
            source_world=source_world_instance,
            target_world=target_world_instance,
            strategy="add_new",
        )
        typer.secho(
            f"Successfully merged world '{source_world_instance.name}' into '{target_world_instance.name}'.",
            fg=typer.colors.GREEN,
        )
    except Exception as e:
        typer.secho(
            f"Error during DB-to-DB merge from '{source_world_instance.name}' to '{target_world_instance.name}': {e}",
            fg=typer.colors.RED,
        )
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="split-world-db")
def cli_split_world_db(
    source_world: Annotated[str, typer.Argument(help="Name of the source ECS world to split.")],
    selected_target_world: Annotated[str, typer.Argument(help="Name of the target world for selected entities.")],
    remaining_target_world: Annotated[str, typer.Argument(help="Name of the target world for remaining entities.")],
    component_name: Annotated[
        str, typer.Option(..., "--component-name", "-cn", help="Name of the component for criteria.")
    ],
    attribute_name: Annotated[str, typer.Option(..., "--attribute", "-a", help="Attribute name in the component.")],
    attribute_value: Annotated[str, typer.Option(..., "--value", "-v", help="Value to match for the attribute.")],
    operator: Annotated[
        str,
        typer.Option(
            "--operator",
            "-op",
            help="Comparison operator: eq, ne, contains, startswith, endswith, gt, lt, ge, le.",
        ),
    ] = "eq",
    delete_from_source: Annotated[bool, typer.Option(help="Delete entities from source world after copying.")] = False,
):
    """
    Splits entities from a source ECS world into two target ECS worlds (DB-to-DB)
    based on a component attribute criterion.
    """
    typer.echo(
        f"Splitting world '{source_world}' into '{selected_target_world}' (selected) and '{remaining_target_world}' (remaining)."
    )
    typer.echo(f"Criteria: {component_name}.{attribute_name} {operator} '{attribute_value}'")
    if delete_from_source:
        typer.confirm(
            f"WARNING: This will delete entities from the source world '{source_world}' after copying. Are you sure?",
            abort=True,
        )

    # Validate worlds
    try:
        app_config.settings.get_world_config(source_world)
        app_config.settings.get_world_config(selected_target_world)
        app_config.settings.get_world_config(remaining_target_world)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if (
        source_world == selected_target_world
        or source_world == remaining_target_world
        or selected_target_world == remaining_target_world
    ):
        typer.secho("Error: Source and target worlds must all be unique for split operation.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Type conversion for attribute_value might be needed depending on component attribute type
    # For now, passing as string. Service layer might need to handle type casting.
    typed_criteria_value: Any = attribute_value
    # TODO: Add type casting logic here if necessary based on component model inspection or hints.

    source_world_inst = get_world(source_world)
    selected_target_inst = get_world(selected_target_world)
    remaining_target_inst = get_world(remaining_target_world)

    if not source_world_inst:
        typer.secho(f"Error: Source world '{source_world}' not found or not initialized.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not selected_target_inst:
        typer.secho(
            f"Error: Selected target world '{selected_target_world}' not found or not initialized.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)
    if not remaining_target_inst:
        typer.secho(
            f"Error: Remaining target world '{remaining_target_world}' not found or not initialized.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        # Service function now takes World objects and handles its own sessions
        count_selected, count_remaining = world_service.split_ecs_world(
            source_world=source_world_inst,
            target_world_selected=selected_target_inst,
            target_world_remaining=remaining_target_inst,
            criteria_component_name=component_name,
            criteria_component_attr=attribute_name,
            criteria_value=typed_criteria_value,
            criteria_op=operator,
            delete_from_source=delete_from_source,
        )
        typer.secho(
            f"Split complete: {count_selected} entities to '{selected_target_inst.name}', "
            f"{count_remaining} entities to '{remaining_target_inst.name}'.",
            fg=typer.colors.GREEN,
        )
        if delete_from_source:
            typer.secho(f"Entities deleted from source world '{source_world_inst.name}'.", fg=typer.colors.YELLOW)

    except Exception as e:
        typer.secho(f"Error during DB-to-DB split from '{source_world_inst.name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


# if __name__ == "__main__":
# Clear any worlds from previous test runs or states if CLI is run multiple times in a process
# For typical CLI execution, this is not strictly necessary as it's a new process each time.
# However, if used programmatically or in tests where the module might be re-imported/re-used:
# clear_world_registry() # Moved to run_cli_directly
# app()


def run_cli_directly():
    """Function to run the CLI when script is executed directly."""
    # Clear registry specifically for direct script runs, not for imports by tests.
    clear_world_registry()
    app()


# --- Transcoding Commands ---
transcode_app = AsyncTyper(name="transcode", help="Manage transcoding profiles and operations.")
app.add_typer(transcode_app)


@transcode_app.command("profile-create")
async def cli_transcode_profile_create(  # Made async
    ctx: typer.Context,
    profile_name: Annotated[str, typer.Option("--name", "-n", help="Unique name for the transcode profile.")],
    tool_name: Annotated[str, typer.Option("--tool", "-t", help="Transcoding tool (e.g., ffmpeg, cjxl).")],
    parameters: Annotated[
        str, typer.Option("--params", "-p", help="Tool parameters. Use {input} and {output} placeholders.")
    ],
    output_format: Annotated[
        str, typer.Option("--format", "-f", help="Target output format/extension (e.g., avif, jxl, mp4).")
    ],
    description: Annotated[Optional[str], typer.Option("--desc", help="Optional description for the profile.")] = None,
):
    """Creates a new transcoding profile."""
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    async def _create():
        typer.echo("DEBUG: Entering _create for profile-create")  # Debug print
        try:
            profile_entity = await transcode_service.create_transcode_profile(
                world=target_world,
                profile_name=profile_name,
                tool_name=tool_name,
                parameters=parameters,
                output_format=output_format,
                description=description,
            )
            typer.secho(
                f"Transcode profile '{profile_name}' (Entity ID: {profile_entity.id}) created successfully in world '{target_world.name}'.",
                fg=typer.colors.GREEN,
            )
        except transcode_service.TranscodeServiceError as e:
            typer.secho(f"Error creating profile: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(f"Unexpected error creating profile: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            raise typer.Exit(code=1)

    await _create()  # Await async call


@transcode_app.command("apply")
async def cli_transcode_apply(  # Made async
    ctx: typer.Context,
    asset_identifier: Annotated[
        str, typer.Option("--asset", "-a", help="Entity ID or SHA256 hash of the source asset.")
    ],
    profile_identifier: Annotated[
        str, typer.Option("--profile", "-p", help="Entity ID or name of the transcode profile.")
    ],
    output_path_str: Annotated[
        Optional[str],
        typer.Option(
            "--output-dir",
            "-o",
            help="Optional parent directory for the transcoded file (mainly for external storage). If not set, uses DAM internal storage via ingestion.",
        ),
    ] = None,
):
    """Applies a transcode profile to an asset."""
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    output_dir: Optional[Path] = Path(output_path_str) if output_path_str else None
    if output_dir and not output_dir.is_dir():
        typer.secho(
            f"Error: Specified output directory '{output_dir}' does not exist or is not a directory.",
            fg=typer.colors.RED,
        )
        # It might be better to create it in the service, but for CLI, explicit existence is safer.
        # Or, the service's `transcode_media` creates it. Let's assume service handles creation.
        # For now, let's remove this check and let the service layer handle path logic.
        # raise typer.Exit(code=1)
        pass

    async def _apply():
        from dam.services import ecs_service  # Placed here to avoid circular import issues at top level with models

        source_asset_entity_id: Optional[int] = None
        async with target_world.db_session_maker() as session:
            try:
                source_asset_entity_id = int(asset_identifier)
            except ValueError:  # Not an int, assume it's a SHA256 hash
                typer.echo(
                    f"Asset identifier '{asset_identifier}' is not an ID, attempting to resolve as SHA256 hash..."
                )
                entity_id_from_hash = await ecs_service.find_entity_id_by_hash(
                    session, hash_value=asset_identifier, hash_type="sha256"
                )
                if not entity_id_from_hash:
                    typer.secho(f"Error: No asset found with SHA256 hash '{asset_identifier}'.", fg=typer.colors.RED)
                    raise typer.Exit(code=1)
                source_asset_entity_id = entity_id_from_hash
                typer.echo(f"Resolved SHA256 hash '{asset_identifier}' to Entity ID: {source_asset_entity_id}")

            if source_asset_entity_id is None:  # Should not happen if logic above is correct
                typer.secho(
                    f"Error: Could not determine source asset entity ID from '{asset_identifier}'.", fg=typer.colors.RED
                )
                raise typer.Exit(code=1)

            profile_id_to_use: int | str
            try:
                profile_id_to_use = int(profile_identifier)
            except ValueError:
                profile_id_to_use = profile_identifier  # Use as name

            try:
                typer.echo(
                    f"Applying transcode profile '{profile_identifier}' to asset ID {source_asset_entity_id} in world '{target_world.name}'..."
                )
                transcoded_entity = await transcode_service.apply_transcode_profile(
                    world=target_world,
                    source_asset_entity_id=source_asset_entity_id,
                    profile_entity_id=profile_id_to_use,  # Pass as int or str
                    output_parent_dir=output_dir,
                )
                typer.secho(
                    f"Transcoding successful. New transcoded asset Entity ID: {transcoded_entity.id}.",
                    fg=typer.colors.GREEN,
                )
                # Optionally display some info about the new asset
                from dam.models.properties.file_properties_component import (
                    FilePropertiesComponent,
                )  # Ensure type is imported

                new_fpc = await ecs_service.get_component(session, transcoded_entity.id, FilePropertiesComponent)
                if new_fpc:
                    typer.echo(f"  New Filename: {new_fpc.original_filename}, Size: {new_fpc.file_size_bytes} bytes")  # type: ignore

            except transcode_service.TranscodeServiceError as e:
                typer.secho(f"Error applying transcode profile: {e}", fg=typer.colors.RED)
                # Check if it's a TranscodeError from media_utils for more specific advice
                if isinstance(e.__cause__, TranscodeError):
                    if "Command not found" in str(e.__cause__) or "not found in PATH" in str(e.__cause__):
                        typer.secho(
                            "Hint: Ensure the required transcoding tool (e.g., ffmpeg, cjxl) is installed and accessible in your system's PATH.",
                            fg=typer.colors.YELLOW,
                        )
                raise typer.Exit(code=1)
            except Exception as e:
                typer.secho(f"Unexpected error during transcoding: {e}", fg=typer.colors.RED)
                typer.secho(traceback.format_exc(), fg=typer.colors.RED)
                raise typer.Exit(code=1)

    await _apply()  # Await async call


# --- Evaluation Commands ---
eval_app = AsyncTyper(name="evaluate", help="Manage and run transcoding evaluations.")
app.add_typer(eval_app)


@eval_app.command("run-create")
async def cli_eval_run_create(  # Made async
    ctx: typer.Context,
    run_name: Annotated[str, typer.Option("--name", "-n", help="Unique name for the evaluation run.")],
    description: Annotated[Optional[str], typer.Option("--desc", help="Optional description for the run.")] = None,
):
    """Creates a new evaluation run concept."""
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        run_entity = await evaluation_systems.create_evaluation_run_concept(
            world=target_world, run_name=run_name, description=description
        )
        typer.secho(
            f"Evaluation run '{run_name}' (Entity ID: {run_entity.id}) created successfully in world '{target_world.name}'.",
            fg=typer.colors.GREEN,
        )
    except evaluation_systems.EvaluationError as e:
        typer.secho(f"Error creating evaluation run: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@eval_app.command("run-execute")
async def cli_eval_run_execute(  # Made async
    ctx: typer.Context,
    run_identifier: Annotated[str, typer.Option("--run", "-r", help="Name or Entity ID of the evaluation run.")],
    asset_identifiers_str: Annotated[
        str, typer.Option("--assets", "-a", help="Comma-separated list of source asset Entity IDs or SHA256 hashes.")
    ],
    profile_identifiers_str: Annotated[
        str, typer.Option("--profiles", "-p", help="Comma-separated list of transcode profile Entity IDs or names.")
    ],
):
    """Executes a pre-defined evaluation run."""
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    asset_identifiers = [item.strip() for item in asset_identifiers_str.split(",")]
    profile_identifiers = [item.strip() for item in profile_identifiers_str.split(",")]

    # Attempt to convert numeric identifiers to int
    typed_asset_ids: List[Union[int, str]] = []
    for aid in asset_identifiers:
        try:
            typed_asset_ids.append(int(aid))
        except ValueError:
            typed_asset_ids.append(aid)

    typed_profile_ids: List[Union[int, str]] = []
    for pid in profile_identifiers:
        try:
            typed_profile_ids.append(int(pid))
        except ValueError:
            typed_profile_ids.append(pid)

    run_id_to_use: Union[int, str]
    try:
        run_id_to_use = int(run_identifier)
    except ValueError:
        run_id_to_use = run_identifier

    async def _execute():
        try:
            typer.echo(f"Executing evaluation run '{run_identifier}' in world '{target_world.name}'...")
            results = await evaluation_systems.execute_evaluation_run(
                world=target_world,
                evaluation_run_id_or_name=run_id_to_use,
                source_asset_identifiers=typed_asset_ids,
                profile_identifiers=typed_profile_ids,
            )
            typer.secho(
                f"Evaluation run '{run_identifier}' completed. Generated {len(results)} results.",
                fg=typer.colors.GREEN,
            )
            if not results:
                typer.echo(
                    "No results were generated. Check logs for details on skipped items or errors during processing."
                )
            # Optionally, print a summary of results here or direct to use 'evaluate report'
        except evaluation_systems.EvaluationError as e:
            typer.secho(f"Error executing evaluation run: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            raise typer.Exit(code=1)

    await _execute()  # Await async call


@eval_app.command("report")
async def cli_eval_report(  # Made async
    ctx: typer.Context,
    run_identifier: Annotated[
        str, typer.Option("--run", "-r", help="Name or Entity ID of the evaluation run to report on.")
    ],
):
    """Displays a report for a completed evaluation run."""
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    run_id_to_use: Union[int, str]
    try:
        run_id_to_use = int(run_identifier)
    except ValueError:
        run_id_to_use = run_identifier

    async def _report():
        try:
            results_data = await evaluation_systems.get_evaluation_results(
                world=target_world, evaluation_run_id_or_name=run_id_to_use
            )
            if not results_data:
                typer.secho(f"No results found for evaluation run '{run_identifier}'.", fg=typer.colors.YELLOW)
                return

            typer.secho(
                f"\n--- Evaluation Report for Run: '{results_data[0]['evaluation_run_name']}' ---", bold=True
            )  # Assumes all results have same run_name

            # Simple table print using Typer/Rich echo. For complex tables, consider `rich.table.Table`.
            # Headers
            headers = [
                "Orig. Asset ID",
                "Orig. Filename",
                "Profile",
                "Tool",
                "Params",
                "Format",
                "Transcoded ID",
                "Transcoded Filename",
                "Size (Bytes)",
                "VMAF",
                "SSIM",
                "PSNR",
                "Custom Metrics",
                "Notes",
            ]
            # typer.echo("| " + " | ".join(headers) + " |") # Basic header
            # For better formatting, print each result.

            for res in results_data:
                typer.echo("---")
                typer.echo(
                    f"  Original Asset: {res['original_asset_filename']} (ID: {res['original_asset_entity_id']})"
                )
                typer.echo(
                    f"  Profile: {res['profile_name']} (Tool: {res['profile_tool']}, Format: {res['profile_format']})"
                )
                typer.echo(f"    Params: {res['profile_params']}")
                typer.echo(
                    f"  Transcoded Asset: {res['transcoded_asset_filename']} (ID: {res['transcoded_asset_entity_id']})"
                )
                typer.echo(f"    File Size: {res['file_size_bytes']} bytes")
                typer.echo(f"    VMAF: {res['vmaf_score'] if res['vmaf_score'] is not None else 'N/A'}")
                typer.echo(f"    SSIM: {res['ssim_score'] if res['ssim_score'] is not None else 'N/A'}")
                typer.echo(f"    PSNR: {res['psnr_score'] if res['psnr_score'] is not None else 'N/A'}")
                if res["custom_metrics"]:
                    typer.echo(f"    Custom Metrics: {json.dumps(res['custom_metrics'])}")
                if res["notes"]:
                    typer.echo(f"    Notes: {res['notes']}")
            typer.echo("---\nReport End.")

        except evaluation_systems.EvaluationError as e:
            typer.secho(f"Error generating report: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            raise typer.Exit(code=1)

    await _report()  # Await async call


@app.command(name="ui")
def cli_ui(ctx: typer.Context):
    """
    Launches the PyQt6 UI for the Digital Asset Management system.
    """
    typer.echo("Launching DAM UI...")
    try:
        import sys

        from PyQt6.QtWidgets import QApplication

        from dam.ui.main_window import MainWindow

        # Check if a QApplication instance already exists.
        # This is important if the CLI is run multiple times in the same process (e.g., testing)
        # or if other parts of the application might also create a QApplication.
        app_instance = QApplication.instance()
        if app_instance is None:
            app_instance = QApplication(sys.argv)
        else:
            typer.echo("Reusing existing QApplication instance.")

        # Get the current world context to pass to the UI
        current_world_instance = None
        if global_state.world_name:
            current_world_instance = get_world(global_state.world_name)

        if not current_world_instance:
            # This case should ideally be handled by main_callback for most commands,
            # but for UI, we might want to allow launching even if no world is immediately active,
            # or prompt the user. For now, let's be strict.
            typer.secho(
                f"Error: Cannot launch UI. Active world '{global_state.world_name}' not found or no world selected.",
                fg=typer.colors.RED,
            )
            typer.echo("Use --world <world_name> or ensure a default world is configured.")
            raise typer.Exit(code=1)

        typer.echo(f"Launching UI with world context: '{current_world_instance.name}'")
        main_window = MainWindow(current_world=current_world_instance)
        main_window.show()

        # Start the Qt event loop.
        # sys.exit(app_instance.exec()) # This would exit the CLI.
        # For a CLI command that launches a GUI, we usually want the GUI to run
        # and the CLI command to finish, allowing the GUI to operate independently
        # or for the CLI to return to the prompt.
        # If the GUI is modal or the main point, app_instance.exec() is correct.
        # If it's a non-blocking launch, we might not call exec() here,
        # or ensure it doesn't block the CLI from exiting.
        # For now, let's assume we want the UI to run and the CLI to wait.
        exit_code = app_instance.exec()
        sys.exit(exit_code)

    except ImportError as e:
        if "PyQt6" in str(e):
            typer.secho(
                "Error: PyQt6 is not installed. Please install it to use the UI.",
                fg=typer.colors.RED,
            )
            typer.echo("You can typically install it using: pip install ecs-dam-system[ui]")
            typer.echo(f"Details: {e}")
        else:
            typer.secho(f"Error launching UI: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred while launching the UI: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    run_cli_directly()


# --- Character Commands ---
character_app = AsyncTyper(name="character", help="Manage character concepts and their links to assets.")
app.add_typer(character_app)

@character_app.command("create")
async def cli_character_create(
    ctx: typer.Context,
    name: Annotated[str, typer.Option("--name", "-n", help="Unique name for the character concept.")],
    description: Annotated[Optional[str], typer.Option("--desc", "-d", help="Optional description for the character.")] = None,
):
    """Creates a new character concept."""
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    async with target_world.db_session_maker() as session:
        try:
            char_entity = await character_service.create_character_concept(
                session=session,
                name=name,
                description=description,
            )
            if char_entity:
                typer.secho(
                    f"Character concept '{name}' (Entity ID: {char_entity.id}) created successfully in world '{target_world.name}'.",
                    fg=typer.colors.GREEN,
                )
                await session.commit() # Commit the transaction
            else:
                # This case might happen if the character already exists and create_character_concept returns it
                # Or if there was an error logged by the service but no exception raised to here.
                # Check service logs for more details.
                typer.secho(f"Character concept '{name}' might already exist or could not be created. Check logs.", fg=typer.colors.YELLOW)

        except ValueError as e: # From service if name is empty
            typer.secho(f"Error: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        except Exception as e_inner: # Catch any other exception during the service call
            typer.secho(f"INNER EXCEPTION in character create: {type(e_inner).__name__}: {e_inner}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(f"Unexpected error creating character concept: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            raise typer.Exit(code=1)


@character_app.command("apply")
async def cli_character_apply(
    ctx: typer.Context,
    asset_identifier: Annotated[str, typer.Option("--asset", "-a", help="Entity ID or SHA256 hash of the asset to link.")],
    character_identifier: Annotated[str, typer.Option("--character", "-c", help="Name or Entity ID of the character concept.")],
    role: Annotated[Optional[str], typer.Option("--role", "-r", help="Optional role of the character in this asset.")] = None,
):
    """Applies (links) a character to an asset."""
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    async with target_world.db_session_maker() as session:
        try:
            # Resolve asset_identifier to entity_id
            asset_entity_id: Optional[int] = None
            try:
                asset_entity_id = int(asset_identifier)
            except ValueError:
                entity_id_from_hash = await dam_ecs_service.find_entity_id_by_hash(
                    session, hash_value=asset_identifier, hash_type="sha256"
                )
                if not entity_id_from_hash:
                    typer.secho(f"Error: No asset found with SHA256 hash '{asset_identifier}'.", fg=typer.colors.RED)
                    raise typer.Exit(code=1)
                asset_entity_id = entity_id_from_hash

            if asset_entity_id is None: # Should be caught above
                raise typer.Exit(code=1)

            # Resolve character_identifier to character_concept_entity_id
            character_concept_entity_id: Optional[int] = None
            try:
                character_concept_entity_id = int(character_identifier)
                # Verify it's a valid character concept
                if not await character_service.get_character_concept_by_id(session, character_concept_entity_id):
                    raise character_service.CharacterConceptNotFoundError
            except ValueError:
                char_concept_entity = await character_service.get_character_concept_by_name(session, character_identifier)
                character_concept_entity_id = char_concept_entity.id
            except character_service.CharacterConceptNotFoundError:
                typer.secho(f"Error: Character concept '{character_identifier}' not found.", fg=typer.colors.RED)
                raise typer.Exit(code=1)

            if character_concept_entity_id is None: # Should be caught
                raise typer.Exit(code=1)

            link_component = await character_service.apply_character_to_entity(
                session=session,
                entity_id_to_link=asset_entity_id,
                character_concept_entity_id=character_concept_entity_id,
                role=role,
            )

            if link_component:
                typer.secho(
                    f"Successfully linked character '{character_identifier}' to asset '{asset_identifier}' with role '{role}'.",
                    fg=typer.colors.GREEN,
                )
                await session.commit() # Commit the transaction
            else:
                # This might happen if the link already exists and the service returns None or the existing link.
                # The service logs a warning in such cases.
                typer.secho(
                    f"Could not link character '{character_identifier}' to asset '{asset_identifier}'. "
                    "It might already be linked with the same role, or an error occurred. Check logs.",
                    fg=typer.colors.YELLOW
                )

        except character_service.CharacterConceptNotFoundError:
             typer.secho(f"Error: Character concept '{character_identifier}' not found.", fg=typer.colors.RED)
             raise typer.Exit(code=1)
        except dam_ecs_service.EntityNotFoundError: # If asset entity ID is invalid and not caught by hash lookup
            typer.secho(f"Error: Asset with identifier '{asset_identifier}' not found.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(f"Unexpected error applying character to asset: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            raise typer.Exit(code=1)


@character_app.command("list-for-asset")
async def cli_character_list_for_asset(
    ctx: typer.Context,
    asset_identifier: Annotated[str, typer.Option("--asset", "-a", help="Entity ID or SHA256 hash of the asset.")],
):
    """Lists all characters linked to a specific asset."""
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    async with target_world.db_session_maker() as session:
        try:
            asset_entity_id: Optional[int] = None
            try:
                asset_entity_id = int(asset_identifier)
            except ValueError:
                entity_id_from_hash = await dam_ecs_service.find_entity_id_by_hash(
                    session, hash_value=asset_identifier, hash_type="sha256"
                )
                if not entity_id_from_hash:
                    typer.secho(f"Error: No asset found with SHA256 hash '{asset_identifier}'.", fg=typer.colors.RED)
                    raise typer.Exit(code=1)
                asset_entity_id = entity_id_from_hash

            if asset_entity_id is None:
                raise typer.Exit(code=1)

            characters_on_asset = await character_service.get_characters_for_entity(session, asset_entity_id)

            if not characters_on_asset:
                typer.secho(f"No characters found for asset '{asset_identifier}'.", fg=typer.colors.YELLOW)
                return

            typer.echo(f"Characters linked to asset '{asset_identifier}' (Entity ID: {asset_entity_id}):")
            for char_concept_entity, role in characters_on_asset:
                char_comp = await dam_ecs_service.get_component(session, char_concept_entity.id, CharacterConceptComponent)
                char_name = char_comp.concept_name if char_comp else "Unknown Character"
                role_str = f" (Role: {role})" if role else ""
                typer.echo(f"  - {char_name} (Concept ID: {char_concept_entity.id}){role_str}")

        except dam_ecs_service.EntityNotFoundError:
             typer.secho(f"Error: Asset with identifier '{asset_identifier}' not found.", fg=typer.colors.RED)
             raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(f"Unexpected error listing characters for asset: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            raise typer.Exit(code=1)


@character_app.command("find-assets")
async def cli_character_find_assets(
    ctx: typer.Context,
    character_identifier: Annotated[str, typer.Option("--character", "-c", help="Name or Entity ID of the character concept.")],
    role_filter: Annotated[Optional[str], typer.Option("--role", "-r", help="Filter by specific role. Use '__NONE__' for assets where character has no role specified, or '__ANY__' for any role.")] = None,
):
    """Finds all assets linked to a specific character."""
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    actual_role_filter = role_filter
    filter_by_role_presence: Optional[bool] = None
    if role_filter == "__NONE__":
        actual_role_filter = None
        filter_by_role_presence = False
    elif role_filter == "__ANY__":
        actual_role_filter = None
        filter_by_role_presence = True


    async with target_world.db_session_maker() as session:
        try:
            character_concept_entity_id: Optional[int] = None
            try:
                character_concept_entity_id = int(character_identifier)
                # Verify it's a valid character concept
                if not await character_service.get_character_concept_by_id(session, character_concept_entity_id):
                    raise character_service.CharacterConceptNotFoundError
            except ValueError:
                char_concept_entity = await character_service.get_character_concept_by_name(session, character_identifier)
                character_concept_entity_id = char_concept_entity.id
            except character_service.CharacterConceptNotFoundError:
                typer.secho(f"Error: Character concept '{character_identifier}' not found.", fg=typer.colors.RED)
                raise typer.Exit(code=1)

            if character_concept_entity_id is None:
                raise typer.Exit(code=1)

            # Fetch character name for display *before* using it in messages
            char_comp_for_display = await dam_ecs_service.get_component(session, character_concept_entity_id, CharacterConceptComponent)
            display_character_name = char_comp_for_display.concept_name if char_comp_for_display else character_identifier # Fallback

            linked_assets = await character_service.get_entities_for_character(
                session,
                character_concept_entity_id,
                role_filter=actual_role_filter,
                filter_by_role_presence=filter_by_role_presence
            )

            # Fetch character name for display
            char_comp_for_display = await dam_ecs_service.get_component(session, character_concept_entity_id, CharacterConceptComponent)
            display_character_name = char_comp_for_display.concept_name if char_comp_for_display else character_identifier # Fallback

            if not linked_assets:
                typer.secho(f"No assets found for character '{display_character_name}' with specified role filter.", fg=typer.colors.YELLOW)
                return

            typer.echo(f"Assets linked to character '{display_character_name}' (Concept ID: {character_concept_entity_id}):")
            for asset_entity in linked_assets:
                fpc = await dam_ecs_service.get_component(session, asset_entity.id, FilePropertiesComponent)
                filename = fpc.original_filename if fpc else "N/A"
                # To show roles, we'd need to query EntityCharacterLinkComponent for each asset or enhance get_entities_for_character
                typer.echo(f"  - Asset ID: {asset_entity.id}, Filename: {filename}")

        except character_service.CharacterConceptNotFoundError:
             typer.secho(f"Error: Character concept '{character_identifier}' not found.", fg=typer.colors.RED)
             raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(f"Unexpected error finding assets for character: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            raise typer.Exit(code=1)

# --- Search Commands ---
search_app = AsyncTyper(name="search", help="Search for assets using various methods.")
app.add_typer(search_app)

@search_app.command("semantic")
async def cli_search_semantic(
    ctx: typer.Context,
    query: Annotated[str, typer.Option("--query", "-q", help="Text query for semantic search.")],
    top_n: Annotated[int, typer.Option("--top-n", "-n", help="Number of top results to return.")] = 10,
    model_name: Annotated[Optional[str], typer.Option("--model", "-m", help="Name of the sentence transformer model to use (optional).")] = None,
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
        model_name=model_name, # Will use service default if None
    )

    typer.echo(f"Dispatching SemanticSearchQuery (Request ID: {request_id}) to world '{target_world.name}' for query: '{query[:100]}...'")

    async def dispatch_and_await_results():
        query_event.result_future = asyncio.get_running_loop().create_future()
        await target_world.dispatch_event(query_event)
        try:
            # Result is List[Tuple[Entity, float, TextEmbeddingComponent]]
            results = await asyncio.wait_for(query_event.result_future, timeout=60.0) # Increased timeout for potentially heavy query

            if not results:
                typer.secho(f"No semantic matches found for query: '{query[:100]}...'. Request ID: {request_id}", fg=typer.colors.YELLOW)
                return

            typer.secho(f"--- Semantic Search Results (Request ID: {request_id}) ---", fg=typer.colors.GREEN)
            typer.echo(f"Found {len(results)} results for query '{query[:100]}...':")
            async with target_world.db_session_maker() as session: # New session for fetching components for display
                for (entity, score, emb_comp) in results:
                    fpc = await dam_ecs_service.get_component(session, entity.id, FilePropertiesComponent)
                    filename = fpc.original_filename if fpc else "N/A"
                    source_info = f"{emb_comp.source_component_name}.{emb_comp.source_field_name}" if emb_comp else "N/A"
                    typer.echo(
                        f"  - Entity ID: {entity.id}, Score: {score:.4f}, Filename: {filename}"
                        f"\n    Matched on: {source_info} (Model: {emb_comp.model_name if emb_comp else 'N/A'})"
                    )
        except asyncio.TimeoutError:
            typer.secho(f"Semantic search query timed out for Request ID: {request_id}.", fg=typer.colors.RED)
            raise typer.Exit(code=1) # Exit on timeout
        except Exception as e:
            typer.secho(f"Semantic search query failed for Request ID: {request_id}. Error: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            raise typer.Exit(code=1) # Exit on other future errors

    try:
        await dispatch_and_await_results()
    except typer.Exit: # Re-raise typer.Exit if it comes from dispatch_and_await_results
        raise
    except Exception as e: # Catch other errors from dispatch_and_await_results setup
        typer.secho(f"Error during semantic search dispatch to world '{target_world.name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)

# Placeholder for item search CLI - to be implemented
@search_app.command("items")
async def cli_search_items(
    ctx: typer.Context,
    text: Annotated[Optional[str], typer.Option("--text", "-t", help="Keyword text to search in filenames/descriptions.")] = None,
    tag: Annotated[Optional[str], typer.Option("--tag", help="Filter by tag name.")] = None,
    character: Annotated[Optional[str], typer.Option("--character", help="Filter by character name or ID.")] = None,
    # Add more filters as needed
):
    """
    Searches for items (assets) based on keywords, tags, characters, etc. (Work In Progress)
    """
    if not global_state.world_name:
        typer.secho("Error: No world selected. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho("Item search CLI is a work in progress.", fg=typer.colors.YELLOW)

    # Implementation notes:
    # 1. Create ItemSearchQuery event in dam.core.events.
    # 2. Create a handler in a new system (e.g., dam.systems.search_systems.py)
    #    - This handler will:
    #      - Get initial set of entities based on text search (e.g., in FilePropertiesComponent.original_filename, or other text fields).
    #        This might require iterating or using DB full-text search if available.
    #      - If tag is provided, get entities for that tag using tag_service.
    #      - If character is provided, get entities for that character using character_service.
    #      - Intersect the sets of entities based on the filters provided.
    #      - Return the final list of matching entities.
    # 3. Update this CLI command to dispatch the ItemSearchQuery and display results.
    pass
