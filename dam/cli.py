from pathlib import Path
from typing import Optional  # For type hinting

import typer
from sqlalchemy import select  # Added for SQLAlchemy 2.0 queries
from typing_extensions import Annotated

from dam.core.database import SessionLocal, create_db_and_tables
from dam.core.logging_config import setup_logging  # Import the setup function
from dam.models import (
    AudioPropertiesComponent,  # New
    ContentHashMD5Component,  # Added for MD5
    ContentHashSHA256Component,  # Added for SHA256
    FileLocationComponent,
    FilePropertiesComponent,
    FramePropertiesComponent,
    ImageDimensionsComponent,  # Added
    # VideoPropertiesComponent,  # Removed
)  # Specific models for query
from dam.services import asset_service, ecs_service, file_operations  # Added ecs_service for get_component

app = typer.Typer(name="dam-cli", help="Digital Asset Management System CLI", add_completion=True)


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    # A callback for the main app, can be used for global setup
    # Initialize logging here
    setup_logging()  # Use default level, or it will pick up DAM_LOG_LEVEL
    # If no command is given, Typer will show help.
    # We can add logic here if needed when no command is run.
    if ctx.invoked_subcommand is None:
        # typer.echo("Initializing DAM CLI application...") # Or some other startup message
        pass


@app.command(name="add-asset")
def cli_add_asset(
    path_str: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to the asset file or directory of asset files.",
            exists=True,
            # file_okay=True, # Handled by custom logic
            # dir_okay=True,  # Handled by custom logic
            readable=True,
            resolve_path=True,  # Resolve to absolute path
        ),
    ],
    no_copy: Annotated[
        bool,
        typer.Option(
            "--no-copy",
            help="Add asset(s) by reference, without copying to DAM storage. "
            "The original file path will be stored. "
            "Note: Hash calculation and metadata extraction will still occur.",
        ),
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option(
            "-r",
            "--recursive",
            help="If path is a directory, process files in subdirectories recursively.",
        ),
    ] = False,
):
    """
    Adds new asset file(s) to the DAM system.
    Calculates content hashes and checks for existing assets with the same content.
    Can process a single file or all files in a directory.
    """
    input_path = Path(path_str)
    files_to_process: list[Path] = []

    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        typer.echo(f"Processing directory: {input_path}")
        if recursive:
            for item in input_path.rglob("*"):  # rglob for recursive
                if item.is_file():
                    files_to_process.append(item)
        else:
            for item in input_path.iterdir():
                if item.is_file():
                    files_to_process.append(item)
        if not files_to_process:
            typer.secho(f"No files found in directory {input_path}", fg=typer.colors.YELLOW)
            raise typer.Exit()
    else:
        typer.secho(f"Error: Path {input_path} is not a file or directory.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    total_files = len(files_to_process)
    typer.echo(f"Found {total_files} file(s) to process.")

    processed_count = 0
    added_count = 0
    linked_count = 0
    error_count = 0

    for filepath in files_to_process:
        processed_count += 1
        typer.echo(f"\nProcessing file {processed_count}/{total_files}: {filepath.name} (from {filepath.parent})")

        try:
            original_filename, size_bytes, mime_type = file_operations.get_file_properties(filepath)
            typer.echo(f"  Properties: Name='{original_filename}', Size={size_bytes} bytes, MIME='{mime_type}'")

            # SHA256 hash is calculated within add_asset_file or add_asset_reference
            # We might display it here for user feedback if needed, but it's also logged by asset_service.

        except FileNotFoundError:  # Should not happen due to exists=True and path resolution
            typer.secho(f"  Error: File not found at {filepath}", fg=typer.colors.RED)
            error_count += 1
            continue
        except IOError as e:
            typer.secho(f"  Error reading file {filepath}: {e}", fg=typer.colors.RED)
            error_count += 1
            continue
        except Exception as e:
            typer.secho(
                f"  An unexpected error occurred during file property gathering for {filepath}: {e}",
                fg=typer.colors.RED,
            )
            error_count += 1
            continue

        db = SessionLocal()
        try:
            if no_copy:
                entity, created_new = asset_service.add_asset_reference(
                    session=db,
                    filepath_on_disk=filepath,  # This is the original path
                    original_filename=original_filename,
                    mime_type=mime_type,
                    size_bytes=size_bytes,
                )
            else:
                entity, created_new = asset_service.add_asset_file(
                    session=db,
                    filepath_on_disk=filepath,
                    original_filename=original_filename,
                    mime_type=mime_type,
                    size_bytes=size_bytes,
                )
            db.commit()
            if created_new:
                added_count += 1
                typer.secho(
                    f"  Successfully added new asset. Entity ID: {entity.id}",
                    fg=typer.colors.GREEN,
                )
            else:
                linked_count += 1
                typer.secho(
                    f"  Asset content already exists or referenced. Linked to Entity ID: {entity.id}",
                    fg=typer.colors.YELLOW,
                )

        except Exception as e:
            db.rollback()

            typer.secho(f"  Database error for {filepath.name}: {type(e).__name__} - {e}", fg=typer.colors.RED)
            # typer.secho(f"  Traceback: {traceback.format_exc()}", fg=typer.colors.RED) # Optional: for more detail
            error_count += 1
        finally:
            db.close()

    typer.echo("\n--- Summary ---")
    typer.echo(f"Total files processed: {processed_count}")
    typer.echo(f"New assets added: {added_count}")
    typer.echo(f"Existing assets linked: {linked_count}")
    typer.echo(f"Errors encountered: {error_count}")
    if error_count > 0:
        typer.secho("Some files could not be processed. Please check the errors above.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="setup-db")
def setup_db():
    """
    Initializes the database and creates all tables.
    Run this once before using other commands if the DB is new.
    """
    try:
        create_db_and_tables()
        typer.secho("Database setup complete.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during database setup: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="find-file-by-hash")
def cli_find_file_by_hash(
    hash_value: Annotated[str, typer.Argument(..., help="The hash value of the file to search for.")],
    hash_type: Annotated[
        str, typer.Option(help="Type of the hash (e.g., 'sha256', 'md5'). Default is 'sha256'.")
    ] = "sha256",
    target_filepath: Annotated[
        Optional[Path],
        typer.Option(
            "--file",
            "-f",
            help="Path to a file. If provided, its hash will be calculated and used for searching.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
):
    """
    Finds and displays information about an asset entity by its content hash (SHA256 or MD5).
    You can either provide a hash value directly or a path to a file to calculate its hash.
    """
    actual_hash_value = hash_value
    actual_hash_type = hash_type.lower()

    if target_filepath:
        typer.echo(f"Calculating {actual_hash_type} hash for file: {target_filepath}...")
        try:
            if actual_hash_type == "sha256":
                actual_hash_value = file_operations.calculate_sha256(target_filepath)
            elif actual_hash_type == "md5":
                # We'll need to implement calculate_md5 in file_operations
                actual_hash_value = file_operations.calculate_md5(target_filepath)
            else:
                error_msg = (
                    f"Unsupported hash type for file calculation: {actual_hash_type}. Supported types: 'sha256', 'md5'."
                )
                typer.secho(error_msg, fg=typer.colors.RED)
                raise typer.Exit(code=1)
            typer.echo(f"Calculated {actual_hash_type} hash: {actual_hash_value}")
        except FileNotFoundError:
            typer.secho(f"Error: File not found at {target_filepath}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(f"Error calculating hash for {target_filepath}: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    elif not actual_hash_value:  # Ensure hash_value is provided if target_filepath is not
        typer.secho("Error: Either a hash value or a file path must be provided.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Querying for asset with {actual_hash_type} hash: {actual_hash_value}")
    db = SessionLocal()
    try:
        # Ensure asset_service.find_entity_by_content_hash can handle different hash types
        entity = asset_service.find_entity_by_content_hash(db, actual_hash_value, actual_hash_type)
        if entity:
            typer.secho(f"Found Entity ID: {entity.id}", fg=typer.colors.CYAN)

            # Display all known content hashes for the entity
            typer.echo("  Content Hashes:")
            sha256_stmt = select(ContentHashSHA256Component).where(ContentHashSHA256Component.entity_id == entity.id)
            sha256_hashes = db.execute(sha256_stmt).scalars().all()
            for ch_comp in sha256_hashes:
                typer.echo(f"    - Type: sha256, Value: {ch_comp.hash_value}")

            md5_stmt = select(ContentHashMD5Component).where(ContentHashMD5Component.entity_id == entity.id)
            md5_hashes = db.execute(md5_stmt).scalars().all()
            for ch_comp in md5_hashes:
                typer.echo(f"    - Type: md5, Value: {ch_comp.hash_value}")

            fpc_stmt = select(FilePropertiesComponent).where(FilePropertiesComponent.entity_id == entity.id)
            fpc = db.execute(fpc_stmt).scalar_one_or_none()
            if fpc:
                typer.echo(
                    f"  File Properties: Name='{fpc.original_filename}', "
                    f"Size={fpc.file_size_bytes}, MIME='{fpc.mime_type}'"
                )

            loc_stmt = select(FileLocationComponent).where(FileLocationComponent.entity_id == entity.id)
            locations = db.execute(loc_stmt).scalars().all()
            if locations:
                typer.echo("  File Locations:")
                for loc in locations:
                    # FileLocationComponent does not have 'filepath'.
                    # It has 'file_identifier' and 'original_filename'.
                    # Displaying original_filename is more user-friendly here.
                    typer.echo(
                        f"    - Original Name: '{loc.original_filename}', "
                        f"Identifier: '{loc.file_identifier}', Storage: '{loc.storage_type}'"
                    )
            else:
                typer.echo("  No file locations found for this entity.")

            # Display multimedia components if they exist

            dimensions_props = ecs_service.get_component(db, entity.id, ImageDimensionsComponent)
            if dimensions_props:
                typer.echo("  Image Dimensions:")
                typer.echo(f"    Width: {dimensions_props.width_pixels}px")
                typer.echo(f"    Height: {dimensions_props.height_pixels}px")

            audio_props = ecs_service.get_component(db, entity.id, AudioPropertiesComponent)
            if audio_props:  # This will show for standalone audio and audio part of video
                typer.echo("  Audio Properties:")
                typer.echo(f"    Duration: {audio_props.duration_seconds}s")
                typer.echo(f"    Codec: {audio_props.codec_name}")
                typer.echo(f"    Sample Rate: {audio_props.sample_rate_hz} Hz")
                typer.echo(f"    Channels: {audio_props.channels}")
                typer.echo(f"    Bit Rate: {audio_props.bit_rate_kbps} kbps")

            frame_props = ecs_service.get_component(db, entity.id, FramePropertiesComponent)
            if frame_props:
                typer.echo("  Animated Frame Properties:")
                typer.echo(f"    Frame Count: {frame_props.frame_count}")
                typer.echo(f"    Nominal Frame Rate: {frame_props.nominal_frame_rate} fps")
                typer.echo(f"    Animation Duration: {frame_props.animation_duration_seconds}s")

        else:
            typer.secho(
                f"No asset found with {hash_type} hash: {hash_value}",
                fg=typer.colors.YELLOW,
            )

    except Exception as e:
        import traceback

        typer.secho(f"Error during query for find-file-by-hash: {type(e).__name__} - {e}", fg=typer.colors.RED)
        typer.secho(f"Traceback: {traceback.format_exc()}", fg=typer.colors.RED)  # Add traceback
        raise typer.Exit(code=1)
    finally:
        db.close()


# The old query-by-hash command is now superseded by find-file-by-hash
# We can remove it or mark it as deprecated if needed. For now, removing.


# Placeholder for the old query_assets command, can be removed or updated later
@app.command(name="query-assets-placeholder", hidden=True)
def query_assets_placeholder(
    component_name: Annotated[str, typer.Option(help="Name of the component to query by.")] = "",
    filter_expression: Annotated[str, typer.Option(help="Filter expression (e.g., 'width>1920').")] = "",
):
    """
    Queries assets based on components and their properties. (Placeholder)
    """
    typer.echo(f"Placeholder: Querying assets with component: {component_name}, filter: {filter_expression}")


@app.command(name="find-similar-images")
def cli_find_similar_images(
    image_filepath_str: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to the image file to find similarities for.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    phash_threshold: Annotated[int, typer.Option(help="Maximum Hamming distance for pHash similarity.")] = 4,
    ahash_threshold: Annotated[int, typer.Option(help="Maximum Hamming distance for aHash similarity.")] = 4,
    dhash_threshold: Annotated[int, typer.Option(help="Maximum Hamming distance for dHash similarity.")] = 4,
):
    """
    Finds and displays information about images similar to the provided image,
    based on perceptual hashes (pHash, aHash, dHash).
    """
    image_filepath = Path(image_filepath_str)
    typer.echo(f"Finding images similar to: {image_filepath.name}")
    typer.echo(f"Similarity thresholds: pHash<={phash_threshold}, aHash<={ahash_threshold}, dHash<={dhash_threshold}")

    db = SessionLocal()
    try:
        similar_entities_info = asset_service.find_entities_by_similar_image_hashes(
            session=db,
            image_path=image_filepath,
            phash_threshold=phash_threshold,
            ahash_threshold=ahash_threshold,
            dhash_threshold=dhash_threshold,
        )

        if similar_entities_info:
            typer.secho(f"Found {len(similar_entities_info)} potentially similar image(s):", fg=typer.colors.CYAN)
            for info in similar_entities_info:
                entity = info["entity"]
                # match_type = info["match_type"] # Variable not used, commented out
                distance = info["distance"]
                matched_hash_type = info["hash_type"]

                typer.echo(f"\n  Entity ID: {entity.id} (Matched by {matched_hash_type}, Distance: {distance})")

                # Display File Properties
                fpc_stmt = select(FilePropertiesComponent).where(FilePropertiesComponent.entity_id == entity.id)
                fpc = db.execute(fpc_stmt).scalar_one_or_none()
                if fpc:
                    typer.echo(
                        f"    File: '{fpc.original_filename}', Size: {fpc.file_size_bytes}, MIME: '{fpc.mime_type}'"
                    )

                # Display File Locations
                loc_stmt = select(FileLocationComponent).where(FileLocationComponent.entity_id == entity.id)
                locations = db.execute(loc_stmt).scalars().all()
                if locations:
                    typer.echo("    Locations:")
                    for loc in locations:
                        typer.echo(
                            f"      - Original Name: '{loc.original_filename}', "
                            f"Identifier: '{loc.file_identifier}', Storage: '{loc.storage_type}'"
                        )
                else:
                    typer.echo("    No file locations found for this entity.")
        else:
            typer.secho("No similar images found based on the criteria.", fg=typer.colors.YELLOW)

    except FileNotFoundError:  # Should be caught by Typer's exists=True on the argument, but good as a fallback.
        typer.secho(f"Error: Image file not found at {image_filepath_str}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except ValueError as ve:  # Raised by asset_service if image processing fails for hashing
        typer.secho(f"Error processing image for similarity search: {ve}", fg=typer.colors.RED)
        raise typer.Exit(code=1)  # Ensure failure exit code
    except Exception as e:
        typer.secho(f"An unexpected error occurred during similarity search: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)  # Ensure failure exit code
    finally:
        db.close()


if __name__ == "__main__":
    app()


@app.command(name="export-world")
def cli_export_world(
    filepath_str: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to export the ECS world JSON file to.",
            file_okay=True,
            dir_okay=False,
            writable=True,  # Ensures the directory is writable, not necessarily the file itself yet
            resolve_path=True,
        ),
    ],
):
    """Exports the entire ECS world (entities and components) to a JSON file."""
    export_path = Path(filepath_str)
    if export_path.is_dir():
        typer.secho(
            f"Error: Export path {export_path} is a directory. Please specify a file path.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)
    if export_path.exists():
        overwrite = typer.confirm(f"File {export_path} already exists. Overwrite?", default=False)
        if not overwrite:
            typer.echo("Export cancelled.")
            raise typer.Exit()

    typer.echo(f"Exporting ECS world to: {export_path}")
    db = SessionLocal()
    try:
        # Ensure world_service is imported
        from dam.services import world_service

        world_service.export_ecs_world_to_json(db, export_path)
        typer.secho(f"ECS world successfully exported to {export_path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during ECS world export: {e}", fg=typer.colors.RED)
        import traceback

        typer.secho(f"Traceback: {traceback.format_exc()}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    finally:
        db.close()


@app.command(name="import-world")
def cli_import_world(
    filepath_str: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to the ECS world JSON file to import.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    merge: Annotated[
        bool,
        typer.Option(
            "--merge",
            help="Merge with existing data. If not set, import will fail if conflicting entity IDs exist.",
        ),
    ] = False,
):
    """Imports an ECS world (entities and components) from a JSON file."""
    import_path = Path(filepath_str)
    typer.echo(f"Importing ECS world from: {import_path}")
    if merge:
        typer.echo("Merge mode enabled: Existing entities with same IDs may be updated.")

    db = SessionLocal()
    try:
        # Ensure world_service is imported
        from dam.services import world_service

        world_service.import_ecs_world_from_json(db, import_path, merge=merge)
        typer.secho(f"ECS world successfully imported from {import_path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during ECS world import: {e}", fg=typer.colors.RED)
        import traceback

        typer.secho(f"Traceback: {traceback.format_exc()}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    finally:
        db.close()
