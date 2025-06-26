# --- Framework Imports for Systems ---
import asyncio
import traceback  # Import traceback for detailed error logging
from pathlib import Path
from typing import Any, List, Optional

import typer
from typing_extensions import Annotated

from dam.core import config as app_config
from dam.core.config import WorldConfig # Added
from dam.core.world import ( # Added
    World,
    create_and_register_world,
    get_world,
    get_default_world,
    get_all_registered_worlds,
    create_and_register_all_worlds_from_settings,
    clear_world_registry,
)
from dam.core.logging_config import setup_logging
# ResourceManager and WorldScheduler will be part of World instances
# from dam.core.resources import FileOperationsResource, ResourceManager # Removed
# from dam.core.systems import WorldContext, WorldScheduler # Removed
from dam.core.stages import SystemStage # Keep for system execution if needed directly by CLI
from dam.core.system_params import WorldContext # Keep for system execution
from dam.models import (
    AudioPropertiesComponent,
    ContentHashMD5Component,
    ContentHashSHA256Component,
    FileLocationComponent,
    FilePropertiesComponent,
    FramePropertiesComponent,
    ImageDimensionsComponent,
)
from dam.services import asset_service, ecs_service, file_operations, world_service  # Added world_service

# Ensure all system modules are imported so @system decorators run and register systems globally
# These will be used by WorldScheduler instances within each World.
# Example:
# from dam import systems as _ # Assuming systems are in dam.systems and __init__.py imports them

# No global ResourceManager or WorldScheduler here anymore.
# They are instantiated per World.

app = typer.Typer(name="dam-cli", help="Digital Asset Management System CLI", add_completion=True)


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
    try:
        create_and_register_all_worlds_from_settings()
        # typer.echo(f"Initialized {len(get_all_registered_worlds())} worlds.") # Optional debug
    except Exception as e:
        # If basic world loading from settings fails, it's a critical startup error.
        typer.secho(f"Critical error: Could not initialize worlds from settings: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


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
            if ctx.invoked_subcommand: # Only print if a subcommand is being invoked
                typer.echo(f"Operating on world: '{global_state.world_name}'")
        else:
            # This means the name was determined, but no such world is registered (e.g., config error for that specific world)
            if ctx.invoked_subcommand not in ["list-worlds"]: # Allow list-worlds
                typer.secho(
                    f"Error: World '{global_state.world_name}' is specified or default, but it's not correctly configured or registered.",
                    fg=typer.colors.RED
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
        current_selection_info = f"Current default/selected world: '{global_state.world_name}' (Use --world to change)" \
                                 if global_state.world_name else \
                                 "No default world selected. Use --world <world_name> or see 'list-worlds'."

        typer.echo(current_selection_info)

        if not get_all_registered_worlds() and not app_config.settings.worlds: # Check both raw config and successfully registered
            typer.secho("No DAM worlds seem to be configured. Please set DAM_WORLDS_CONFIG.", fg=typer.colors.YELLOW)


@app.command(name="list-worlds")
def cli_list_worlds():
    """Lists all configured and registered ECS worlds."""
    # Worlds should have been registered by the main_callback's call to create_and_register_all_worlds_from_settings()
    registered_worlds = get_all_registered_worlds()

    if not registered_worlds:
        typer.secho("No ECS worlds are currently registered or configured correctly.", fg=typer.colors.YELLOW)
        typer.echo("Check your DAM_WORLDS_CONFIG environment variable (JSON string or file path) and application logs for errors.")
            return

        typer.echo("Available ECS worlds:")
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
        elif registered_worlds: # Worlds exist, but no default was determined by settings
             typer.secho(
                "\nNote: No specific default world is set. Operations might require explicit --world.",
                fg=typer.colors.YELLOW,
            )
    except Exception as e: # Catch any unexpected errors during listing
        typer.secho(f"Error listing worlds: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="add-asset")
def cli_add_asset(
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
        typer.secho(f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED)
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
    else:  # Should be caught by exists=True
        typer.secho(f"Error: Path {input_path} is not a file or directory.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    total_files = len(files_to_process)
    typer.echo(f"Found {total_files} file(s) to process for world '{global_state.world_name}'.")

    processed_count, added_count, linked_count, error_count = 0, 0, 0, 0

    for filepath in files_to_process:
        processed_count += 1
        typer.echo(
            f"\nProcessing file {processed_count}/{total_files}: {filepath.name} (world: '{global_state.world_name}')"
        )
        try:
            original_filename, size_bytes, mime_type = file_operations.get_file_properties(filepath)
        except Exception as e:
            typer.secho(f"  Error getting properties for {filepath}: {e}", fg=typer.colors.RED)
            error_count += 1
            continue

        db_session = None # Initialize for finally block
        try:
            db_session = target_world.get_db_session()
            if no_copy:
                entity, created_new = asset_service.add_asset_reference(
                    session=db_session,
                    filepath_on_disk=filepath,
                    original_filename=original_filename,
                    mime_type=mime_type,
                    size_bytes=size_bytes,
                    # world_name is removed from service function
                )
            else:
                entity, created_new = asset_service.add_asset_file(
                    session=db_session,
                    filepath_on_disk=filepath,
                    original_filename=original_filename,
                    mime_type=mime_type,
                    size_bytes=size_bytes,
                    world_config=target_world.config, # Pass WorldConfig
                )
            db_session.commit() # Commit changes for this asset

            if created_new:
                added_count += 1
                typer.secho(f"  Successfully added new asset. Entity ID: {entity.id}", fg=typer.colors.GREEN)
            else:
                linked_count += 1
                typer.secho(
                    f"  Asset content already exists/referenced. Linked to Entity ID: {entity.id}",
                    fg=typer.colors.YELLOW,
                )

            # Run system stages using the World's execute_stage method
            # This method handles session creation and management internally for the stage.
            typer.echo(
                f"  Running post-processing systems for entity {entity.id} in world '{target_world.name}'..."
            )
            async def run_stages_for_asset():
                # No need to pass session to world.execute_stage, it will manage its own.
                await target_world.execute_stage(SystemStage.METADATA_EXTRACTION)
                # If there were other stages:
                # await target_world.execute_stage(SystemStage.ASSET_POST_PROCESSING)

            asyncio.run(run_stages_for_asset())
            typer.secho(f"  Post-processing systems completed for entity {entity.id}.", fg=typer.colors.GREEN)

        except Exception as e:
            if db_session: # Rollback if session was active
                db_session.rollback()
            typer.secho(f"  Error processing file {filepath.name}: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)
            error_count += 1
        finally:
            if db_session:
                db_session.close()

    typer.echo("\n--- Summary ---")
    typer.echo(f"World: '{target_world.name}'")
    typer.echo(f"Total files processed: {processed_count}")
    typer.echo(f"New assets added: {added_count}")
    typer.echo(f"Existing assets linked: {linked_count}")
    typer.echo(f"Errors encountered: {error_count}")
    if error_count > 0:
        typer.secho("Some files could not be processed. Check errors above.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="setup-db")
def setup_db(ctx: typer.Context):  # Added context
    """
    Initializes the database and creates tables for the specified/default ECS world.
    """
    if not global_state.world_name:
        typer.secho("Error: No world selected for setup-db. Use --world <world_name>.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    target_world = get_world(global_state.world_name)
    if not target_world:
        typer.secho(f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Setting up database for world: '{target_world.name}'...")
    try:
        target_world.create_db_and_tables() # World method calls its db_manager
        typer.secho(f"Database setup complete for world '{target_world.name}'.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during database setup for world '{target_world.name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="find-file-by-hash")
def cli_find_file_by_hash(
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
        typer.secho(f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED)
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

    typer.echo(
        f"Querying world '{target_world.name}' for asset with {actual_hash_type} hash: {actual_hash_value}"
    )
    db_session = None
    try:
        db_session = target_world.get_db_session()
        entity = asset_service.find_entity_by_content_hash(db_session, actual_hash_value, actual_hash_type)
        if entity:
            typer.secho(f"Found Entity ID: {entity.id} in world '{target_world.name}'", fg=typer.colors.CYAN)

            fpc = ecs_service.get_component(db_session, entity.id, FilePropertiesComponent)
            if fpc:
                typer.echo(
                    f"  File Properties: Name='{fpc.original_filename}', Size={fpc.file_size_bytes}, MIME='{fpc.mime_type}'"
                )

            typer.echo("  Content Hashes:")
            sha256_comps = ecs_service.get_components(db_session, entity.id, ContentHashSHA256Component)
            for ch_comp in sha256_comps:
                typer.echo(f"    - Type: sha256, Value: {ch_comp.hash_value}")
            md5_comps = ecs_service.get_components(db_session, entity.id, ContentHashMD5Component)
            for ch_comp in md5_comps:
                typer.echo(f"    - Type: md5, Value: {ch_comp.hash_value}")

            locations = ecs_service.get_components(db_session, entity.id, FileLocationComponent)
            if locations:
                typer.echo("  File Locations:")
                for loc in locations:
                    typer.echo(
                        f"    - Contextual Name: '{loc.contextual_filename}', Content ID: '{loc.content_identifier[:12]}...', Path/Key: '{loc.physical_path_or_key}', Storage: '{loc.storage_type}'"
                    )

            dimensions_props = ecs_service.get_component(db_session, entity.id, ImageDimensionsComponent)
            if dimensions_props:
                typer.echo(
                    f"  Image Dimensions: {dimensions_props.width_pixels}x{dimensions_props.height_pixels}px"
                )

            audio_props = ecs_service.get_component(db_session, entity.id, AudioPropertiesComponent)
            if audio_props:
                typer.echo("  Audio Properties:")
                typer.echo(
                    f"    Duration: {audio_props.duration_seconds}s, Codec: {audio_props.codec_name}, Rate: {audio_props.sample_rate_hz}Hz"
                )

            frame_props = ecs_service.get_component(db_session, entity.id, FramePropertiesComponent)
            if frame_props:
                typer.echo("  Animated Frame Properties:")
                typer.echo(
                    f"    Frames: {frame_props.frame_count}, Rate: {frame_props.nominal_frame_rate}fps, Duration: {frame_props.animation_duration_seconds}s"
                )
        else:
            typer.secho(
                f"No asset found in world '{target_world.name}' with {actual_hash_type} hash: {actual_hash_value}",
                fg=typer.colors.YELLOW,
            )
    except Exception as e:
        typer.secho(f"Error querying in world '{target_world.name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    finally:
        if db_session:
            db_session.close()


@app.command(name="find-similar-images")
def cli_find_similar_images(
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
        typer.secho(f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    image_filepath = Path(image_filepath_str)
    typer.echo(f"Finding images similar to: {image_filepath.name} in world '{target_world.name}'")

    db_session = None
    try:
        db_session = target_world.get_db_session()
        similar_entities_info = asset_service.find_entities_by_similar_image_hashes(
            session=db_session,
            image_path=image_filepath,
            phash_threshold=phash_threshold,
            ahash_threshold=ahash_threshold,
            dhash_threshold=dhash_threshold,
        )
        if similar_entities_info:
            typer.secho(
                f"Found {len(similar_entities_info)} similar image(s) in world '{target_world.name}':",
                fg=typer.colors.CYAN,
            )
            for info in similar_entities_info:
                entity = info["entity"]
                distance = info["distance"]
                matched_hash_type = info["hash_type"]
                typer.echo(f"\n  Entity ID: {entity.id} (Matched by {matched_hash_type}, Distance: {distance})")
                fpc = ecs_service.get_component(db_session, entity.id, FilePropertiesComponent)
                if fpc:
                    typer.echo(
                        f"    File: '{fpc.original_filename}', Size: {fpc.file_size_bytes}, MIME: '{fpc.mime_type}'"
                    )
                # ... (display other components as needed, using db_session) ...
        else:
            typer.secho(f"No similar images found in world '{target_world.name}'.", fg=typer.colors.YELLOW)
    except ValueError as ve: # Specific error from service for bad image
        typer.secho(
            f"Error processing image for similarity search in world '{target_world.name}': {ve}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(
            f"Unexpected error during similarity search in world '{target_world.name}': {e}", fg=typer.colors.RED
        )
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    finally:
        if db_session:
            db_session.close()


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
        typer.secho(f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED)
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
        typer.secho(f"Error: World '{global_state.world_name}' not found or not initialized correctly.", fg=typer.colors.RED)
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

    if source_world_instance.name == target_world_instance.name: # Compare by actual name from instance
        typer.secho("Error: Source and target worlds cannot be the same for merge operation.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Service function now takes World objects and handles its own sessions
        world_service.merge_ecs_worlds_db_to_db(
            source_world=source_world_instance,
            target_world=target_world_instance,
            strategy="add_new",
        )
        typer.secho(f"Successfully merged world '{source_world_instance.name}' into '{target_world_instance.name}'.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during DB-to-DB merge from '{source_world_instance.name}' to '{target_world_instance.name}': {e}", fg=typer.colors.RED)
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
        typer.secho(f"Error: Selected target world '{selected_target_world}' not found or not initialized.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not remaining_target_inst:
        typer.secho(f"Error: Remaining target world '{remaining_target_world}' not found or not initialized.", fg=typer.colors.RED)
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


if __name__ == "__main__":
    # Clear any worlds from previous test runs or states if CLI is run multiple times in a process
    # For typical CLI execution, this is not strictly necessary as it's a new process each time.
    # However, if used programmatically or in tests where the module might be re-imported/re-used:
    clear_world_registry()
    app()
