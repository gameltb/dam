# --- Framework Imports for Systems ---
import asyncio
import traceback  # Import traceback for detailed error logging
from pathlib import Path
from typing import Any, List, Optional  # Added Any

import typer
from typing_extensions import Annotated

from dam.core import config as app_config  # Changed import

# from dam.core.config import settings  # For accessing default world name if needed
# Use db_manager for session handling
from dam.core.database import db_manager  # Changed from SessionLocal, create_db_and_tables
from dam.core.logging_config import setup_logging
from dam.core.resources import FileOperationsResource, ResourceManager
from dam.core.stages import SystemStage
from dam.core.systems import WorldContext, WorldScheduler  # Assuming WorldContext is also in dam.core.systems
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

# Ensure all system modules are imported so @system decorators run and register systems

# --- Initialize Framework Objects ---
resource_manager = ResourceManager()
# Encapsulate file_operations functions into a resource
# FileOperationsResource might need to be designed to take the module or be a namespace
# For now, assuming FileOperationsResource() correctly wraps/provides access to file_operations functions.
# If FileOperationsResource directly uses functions from file_operations module, it's fine.
resource_manager.add_resource(FileOperationsResource())

world_scheduler = WorldScheduler(resource_manager)
# Systems are registered via @system decorator when their modules (e.g., metadata_systems) are imported.


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

    # Determine the target world name
    # Priority: CLI option > Environment Variable > app_config.settings.DEFAULT_WORLD_NAME
    if world:
        global_state.world_name = world
    elif (
        app_config.settings.DEFAULT_WORLD_NAME
    ):  # app_config.settings.DEFAULT_WORLD_NAME should always exist after pydantic validation
        global_state.world_name = app_config.settings.DEFAULT_WORLD_NAME
    else:
        # This case should ideally not be reached if settings validation works correctly
        # and ensures a default world or DAM_WORLDS is configured.
        # If it does, it means no worlds are configured and no default can be inferred.
        # We should prevent commands from running if no world can be determined.
        # For commands like 'list-worlds', this check might be bypassed.
        if ctx.invoked_subcommand not in ["list-worlds"]:  # Allow list-worlds to run without a default
            typer.secho(
                "Error: No ECS world specified and no default world configured. "
                "Use --world <world_name> or set DAM_DEFAULT_WORLD_NAME.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        # For list-worlds, global_state.world_name can remain None

    if global_state.world_name:
        try:
            # Validate that the chosen world_name is actually configured
            # This implicitly uses app_config.settings.get_world_config which would raise ValueError if invalid
            _ = app_config.settings.get_world_config(global_state.world_name)
            if ctx.invoked_subcommand:  # Only print if a subcommand is being invoked
                typer.echo(f"Operating on world: '{global_state.world_name}'")
        except ValueError as e:
            if ctx.invoked_subcommand not in ["list-worlds"]:  # Don't fail for list-worlds
                typer.secho(f"Error: World '{global_state.world_name}' is not configured: {e}", fg=typer.colors.RED)
                raise typer.Exit(code=1)

    if ctx.invoked_subcommand is None:
        # Typer shows help by default.
        # If a default world is identifiable, we could print it.
        if global_state.world_name:
            typer.echo(f"Current default/selected world: '{global_state.world_name}' (Use --world to change)")
        elif not db_manager.get_all_world_names():
            typer.secho("No DAM worlds configured. Please set DAM_WORLDS_CONFIG.", fg=typer.colors.YELLOW)
        else:
            typer.secho(
                "No default world selected. Use --world <world_name> or list available worlds with 'list-worlds'.",
                fg=typer.colors.YELLOW,
            )


@app.command(name="list-worlds")
def cli_list_worlds():
    """Lists all configured ECS worlds."""
    try:
        world_names = db_manager.get_all_world_names()
        if not world_names:
            typer.secho("No ECS worlds are configured.", fg=typer.colors.YELLOW)
            typer.echo("Configure worlds using the DAM_WORLDS_CONFIG environment variable (JSON string or file path).")
            return

        typer.echo("Available ECS worlds:")
        for name in world_names:
            is_default = app_config.settings.DEFAULT_WORLD_NAME == name
            default_marker = " (default)" if is_default else ""
            typer.echo(f"  - {name}{default_marker}")

        if not app_config.settings.DEFAULT_WORLD_NAME and world_names:
            typer.secho(
                "\nNote: No default world is explicitly set. The first configured world might be used by default if not overridden.",
                fg=typer.colors.YELLOW,
            )
        elif app_config.settings.DEFAULT_WORLD_NAME and app_config.settings.DEFAULT_WORLD_NAME not in world_names:
            typer.secho(
                f"\nWarning: Default world '{app_config.settings.DEFAULT_WORLD_NAME}' is set but not found in parsed configurations!",
                fg=typer.colors.RED,
            )

    except Exception as e:
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
    if not global_state.world_name:  # Ensure a world is selected
        typer.secho("Error: No world selected for add-asset. Use --world <world_name>.", fg=typer.colors.RED)
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

        # Get session for the target world
        try:
            with db_manager.get_db_session(global_state.world_name) as db:
                if no_copy:
                    entity, created_new = asset_service.add_asset_reference(
                        session=db,
                        filepath_on_disk=filepath,
                        original_filename=original_filename,
                        mime_type=mime_type,
                        size_bytes=size_bytes,
                        world_name=global_state.world_name,  # Pass world_name for logging/consistency
                    )
                else:
                    entity, created_new = asset_service.add_asset_file(
                        session=db,
                        filepath_on_disk=filepath,
                        original_filename=original_filename,
                        mime_type=mime_type,
                        size_bytes=size_bytes,
                        world_name=global_state.world_name,  # Pass world_name for file_storage
                    )
                db.commit()
                if created_new:
                    added_count += 1
                    typer.secho(f"  Successfully added new asset. Entity ID: {entity.id}", fg=typer.colors.GREEN)
                else:
                    linked_count += 1
                    typer.secho(
                        f"  Asset content already exists/referenced. Linked to Entity ID: {entity.id}",
                        fg=typer.colors.YELLOW,
                    )

                # After primary transaction is committed, run post-processing systems
                # This needs to happen outside the 'db' session context if that session is closed upon exit.
                # However, the WorldContext for the scheduler will need its own session.
                # The original `db` session from `with db_manager.get_db_session...` is committed and closed here.
                # We need a new context for the scheduler.

            # Systems execution happens *after* the main asset addition transaction is complete.
            typer.echo(
                f"  Scheduling post-processing systems for entity {entity.id} in world '{global_state.world_name}'..."
            )
            try:
                current_world_config = app_config.settings.get_world_config(global_state.world_name)

                async def run_system_stages():
                    # Each stage execution should manage its own session via WorldContext
                    # For METADATA_EXTRACTION stage:
                    with db_manager.get_db_session(global_state.world_name) as stage_session:
                        metadata_world_ctx = WorldContext(
                            session=stage_session,
                            world_name=global_state.world_name,  # type: ignore
                            world_config=current_world_config,
                        )
                        await world_scheduler.execute_stage(SystemStage.METADATA_EXTRACTION, metadata_world_ctx)

                    # Example for another stage, if defined and needed:
                    # with db_manager.get_db_session(global_state.world_name) as stage_session_post:
                    #     post_process_world_ctx = WorldContext(
                    #         session=stage_session_post,
                    #         world_name=global_state.world_name,
                    #         world_config=current_world_config
                    #     )
                    #     await world_scheduler.execute_stage(SystemStage.ASSET_POST_PROCESSING, post_process_world_ctx)

                asyncio.run(run_system_stages())
                typer.secho(f"  Post-processing systems completed for entity {entity.id}.", fg=typer.colors.GREEN)

            except Exception as e_sys:
                typer.secho(f"  Error during system execution for {filepath.name}: {e_sys}", fg=typer.colors.RED)
                typer.secho(traceback.format_exc(), fg=typer.colors.RED)
                error_count += 1  # Count system execution errors as well

        except Exception as e:
            # db.rollback() will be handled by session exiting context manager if commit failed
            typer.secho(f"  Database error for {filepath.name}: {e}", fg=typer.colors.RED)
            typer.secho(traceback.format_exc(), fg=typer.colors.RED)  # More detailed traceback
            error_count += 1
        # finally:
        #     db.close() # Handled by context manager

    typer.echo("\n--- Summary ---")
    typer.echo(f"World: '{global_state.world_name}'")
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

    typer.echo(f"Setting up database for world: '{global_state.world_name}'...")
    try:
        # db_manager.create_db_and_tables uses the world_name from its argument
        db_manager.create_db_and_tables(global_state.world_name)
        typer.secho(f"Database setup complete for world '{global_state.world_name}'.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during database setup for world '{global_state.world_name}': {e}", fg=typer.colors.RED)
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

    actual_hash_value = hash_value_arg
    actual_hash_type = hash_type.lower()

    if target_filepath:
        typer.echo(
            f"Calculating {actual_hash_type} hash for file: {target_filepath} (world '{global_state.world_name}')..."
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
        f"Querying world '{global_state.world_name}' for asset with {actual_hash_type} hash: {actual_hash_value}"
    )
    try:
        with db_manager.get_db_session(global_state.world_name) as db:
            entity = asset_service.find_entity_by_content_hash(db, actual_hash_value, actual_hash_type)
            if entity:
                typer.secho(f"Found Entity ID: {entity.id} in world '{global_state.world_name}'", fg=typer.colors.CYAN)
                # ... (rest of the display logic remains largely the same, ensure db is used for queries)
                # Example for one component type:
                fpc = ecs_service.get_component(db, entity.id, FilePropertiesComponent)  # Pass db
                if fpc:
                    typer.echo(
                        f"  File Properties: Name='{fpc.original_filename}', Size={fpc.file_size_bytes}, MIME='{fpc.mime_type}'"
                    )

                # Display all known content hashes for the entity
                typer.echo("  Content Hashes:")
                sha256_comps = ecs_service.get_components(db, entity.id, ContentHashSHA256Component)
                for ch_comp in sha256_comps:
                    typer.echo(f"    - Type: sha256, Value: {ch_comp.hash_value}")
                md5_comps = ecs_service.get_components(db, entity.id, ContentHashMD5Component)
                for ch_comp in md5_comps:
                    typer.echo(f"    - Type: md5, Value: {ch_comp.hash_value}")

                locations = ecs_service.get_components(db, entity.id, FileLocationComponent)
                if locations:
                    typer.echo("  File Locations:")
                    for loc in locations:
                        typer.echo(
                            f"    - Contextual Name: '{loc.contextual_filename}', Content ID: '{loc.content_identifier[:12]}...', Path/Key: '{loc.physical_path_or_key}', Storage: '{loc.storage_type}'"
                        )

                dimensions_props = ecs_service.get_component(db, entity.id, ImageDimensionsComponent)
                if dimensions_props:
                    typer.echo(
                        f"  Image Dimensions: {dimensions_props.width_pixels}x{dimensions_props.height_pixels}px"
                    )

                audio_props = ecs_service.get_component(db, entity.id, AudioPropertiesComponent)
                if audio_props:
                    typer.echo("  Audio Properties:")
                    typer.echo(
                        f"    Duration: {audio_props.duration_seconds}s, Codec: {audio_props.codec_name}, Rate: {audio_props.sample_rate_hz}Hz"
                    )

                frame_props = ecs_service.get_component(db, entity.id, FramePropertiesComponent)
                if frame_props:
                    typer.echo("  Animated Frame Properties:")
                    typer.echo(
                        f"    Frames: {frame_props.frame_count}, Rate: {frame_props.nominal_frame_rate}fps, Duration: {frame_props.animation_duration_seconds}s"
                    )

            else:
                typer.secho(
                    f"No asset found in world '{global_state.world_name}' with {actual_hash_type} hash: {actual_hash_value}",
                    fg=typer.colors.YELLOW,
                )
    except Exception as e:
        typer.secho(f"Error querying in world '{global_state.world_name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    # finally:
    #     if db: # Handled by context manager
    #         db.close()


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

    image_filepath = Path(image_filepath_str)
    typer.echo(f"Finding images similar to: {image_filepath.name} in world '{global_state.world_name}'")

    try:
        with db_manager.get_db_session(global_state.world_name) as db:
            similar_entities_info = asset_service.find_entities_by_similar_image_hashes(
                session=db,  # Pass the world-specific session
                image_path=image_filepath,
                phash_threshold=phash_threshold,
                ahash_threshold=ahash_threshold,
                dhash_threshold=dhash_threshold,
                # world_name=global_state.world_name, # Removed, service function doesn't take it
            )
            if similar_entities_info:
                typer.secho(
                    f"Found {len(similar_entities_info)} similar image(s) in world '{global_state.world_name}':",
                    fg=typer.colors.CYAN,
                )
                for info in similar_entities_info:
                    entity = info["entity"]
                    distance = info["distance"]
                    matched_hash_type = info["hash_type"]
                    typer.echo(f"\n  Entity ID: {entity.id} (Matched by {matched_hash_type}, Distance: {distance})")
                    fpc = ecs_service.get_component(db, entity.id, FilePropertiesComponent)  # Pass db
                    if fpc:
                        typer.echo(
                            f"    File: '{fpc.original_filename}', Size: {fpc.file_size_bytes}, MIME: '{fpc.mime_type}'"
                        )
                    # ... (display other components as needed, using db session) ...
            else:
                typer.secho(f"No similar images found in world '{global_state.world_name}'.", fg=typer.colors.YELLOW)
    except ValueError as ve:
        typer.secho(
            f"Error processing image for similarity search in world '{global_state.world_name}': {ve}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(
            f"Unexpected error during similarity search in world '{global_state.world_name}': {e}", fg=typer.colors.RED
        )
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    # finally:
    #     if db: # Handled by context manager
    #         db.close()


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

    typer.echo(f"Exporting ECS world '{global_state.world_name}' to: {export_path}")
    try:
        with db_manager.get_db_session(global_state.world_name) as db:
            world_service.export_ecs_world_to_json(db, export_path, world_name_for_log=global_state.world_name)
        typer.secho(f"ECS world '{global_state.world_name}' exported to {export_path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error exporting world '{global_state.world_name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    # finally:
    #     if db: # Handled by context manager
    #         db.close()


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

    import_path = Path(filepath_str)
    typer.echo(f"Importing ECS world from: {import_path} into world '{global_state.world_name}'")
    if merge:
        typer.echo("Merge mode enabled.")

    try:
        with db_manager.get_db_session(global_state.world_name) as db:
            world_service.import_ecs_world_from_json(
                db, import_path, merge=merge, world_name_for_log=global_state.world_name
            )
        typer.secho(f"ECS world imported into '{global_state.world_name}' from {import_path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error importing into world '{global_state.world_name}': {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    # finally:
    #     if db: # Handled by context manager
    #         db.close()


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

    # Validate worlds exist
    try:
        app_config.settings.get_world_config(source_world)
        app_config.settings.get_world_config(target_world)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if source_world == target_world:
        typer.secho("Error: Source and target worlds cannot be the same for merge operation.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        with db_manager.get_db_session(source_world) as source_db, db_manager.get_db_session(target_world) as target_db:
            world_service.merge_ecs_worlds_db_to_db(
                source_session=source_db,
                target_session=target_db,
                source_world_name_for_log=source_world,
                target_world_name_for_log=target_world,
                strategy="add_new",
            )
            # merge_ecs_worlds_db_to_db handles its own commit on target_db
            typer.secho(f"Successfully merged world '{source_world}' into '{target_world}'.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during DB-to-DB merge: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        # Rollback target_db session if an error occurred before its commit in service
        # The service function should handle its own rollback on error before commit.
        # Context managers will ensure sessions are closed.
        raise typer.Exit(code=1)
    # finally: # Handled by context managers
    #     if source_db:
    #         source_db.close()
    #     if target_db:
    #         target_db.close()


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
    # Example: if attribute is integer, criteria_value might need int(attribute_value)
    # This is a simplification for CLI. A more robust solution would inspect component model.
    typed_criteria_value: Any = attribute_value
    # Add type casting logic here if necessary, e.g. for int/float/bool
    # For now, the service layer's getattr will fetch the value, and Python's comparison
    # might work for some types (e.g. int('10') == 10 is False, but int('10') > 5 is True)
    # This needs to be robust in the service or by knowing the type.

    try:
        with (
            db_manager.get_db_session(source_world) as source_s,
            db_manager.get_db_session(selected_target_world) as selected_s,
            db_manager.get_db_session(remaining_target_world) as remaining_s,
        ):
            count_selected, count_remaining = world_service.split_ecs_world(
                source_session=source_s,
                target_session_selected=selected_s,
                target_session_remaining=remaining_s,
                criteria_component_name=component_name,
                criteria_component_attr=attribute_name,
                criteria_value=typed_criteria_value,
                criteria_op=operator,
                delete_from_source=delete_from_source,
                source_world_name_for_log=source_world,
                target_selected_world_name_for_log=selected_target_world,
                target_remaining_world_name_for_log=remaining_target_world,
            )
            # Service function handles its own commits for targets and source (if delete)
            typer.secho(
                f"Split complete: {count_selected} entities to '{selected_target_world}', "
                f"{count_remaining} entities to '{remaining_target_world}'.",
                fg=typer.colors.GREEN,
            )
            if delete_from_source:
                typer.secho(f"Entities deleted from source world '{source_world}'.", fg=typer.colors.YELLOW)

    except Exception as e:
        typer.secho(f"Error during DB-to-DB split: {e}", fg=typer.colors.RED)
        typer.secho(traceback.format_exc(), fg=typer.colors.RED)
        # Service function should handle its own rollbacks.
        # Context managers will ensure sessions are closed.
        raise typer.Exit(code=1)
    # finally: # Handled by context managers
    #     if source_s:
    #         source_s.close()
    #     if selected_s:
    #         selected_s.close()
    #     if remaining_s:
    #         remaining_s.close()


if __name__ == "__main__":
    app()
