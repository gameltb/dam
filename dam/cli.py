from pathlib import Path
from typing import Optional  # For type hinting

import typer
from sqlalchemy import select  # Added for SQLAlchemy 2.0 queries
from typing_extensions import Annotated

from dam.core.database import SessionLocal, create_db_and_tables
from dam.core.logging_config import setup_logging  # Import the setup function
from dam.models import (
    ContentHashMD5Component,  # Added for MD5
    ContentHashSHA256Component,  # Added for SHA256
    FileLocationComponent,
    FilePropertiesComponent,
)  # Specific models for query
from dam.services import asset_service, file_operations

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
    filepath_str: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to the asset file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
):
    """
    Adds a new asset file to the DAM system.
    Calculates its SHA256 hash and checks for existing assets with the same content.
    """
    filepath = Path(filepath_str)
    typer.echo(f"Processing file: {filepath.name}")

    try:
        original_filename, size_bytes, mime_type = file_operations.get_file_properties(filepath)
        typer.echo(f"Properties: Name='{original_filename}', Size={size_bytes} bytes, MIME='{mime_type}'")

        content_hash = file_operations.calculate_sha256(filepath)
        typer.echo(f"SHA256 Hash: {content_hash}")

    except FileNotFoundError:
        typer.secho(f"Error: File not found at {filepath}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except IOError as e:
        typer.secho(f"Error reading file {filepath}: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(
            f"An unexpected error occurred during file processing: {e}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    db = SessionLocal()
    try:
        entity, created_new = asset_service.add_asset_file(
            session=db,
            filepath_on_disk=filepath,
            original_filename=original_filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            # content_hash is now derived by add_asset_file internally
        )
        db.commit()
        if created_new:
            typer.secho(
                f"Successfully added new asset. Entity ID: {entity.id}",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(
                f"Asset content already exists. Linked to Entity ID: {entity.id}",
                fg=typer.colors.YELLOW,
            )

    except Exception as e:
        db.rollback()
        typer.secho(f"Database error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    finally:
        db.close()


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
                    typer.echo(f"    - Path: '{loc.filepath}', Storage: '{loc.storage_type}'")
            else:
                typer.echo("  No file locations found for this entity.")
        else:
            typer.secho(
                f"No asset found with {hash_type} hash: {hash_value}",
                fg=typer.colors.YELLOW,
            )

    except Exception as e:
        typer.secho(f"Error during query: {e}", fg=typer.colors.RED)
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
                        typer.echo(f"      - Path: '{loc.filepath}', Storage: '{loc.storage_type}'")
                else:
                    typer.echo("    No file locations found for this entity.")
        else:
            typer.secho("No similar images found based on the criteria.", fg=typer.colors.YELLOW)

    except FileNotFoundError:  # Should be caught by Typer's exists=True, but good practice
        typer.secho(f"Error: Image file not found at {image_filepath_str}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except ValueError as ve:  # E.g. if image cannot be processed
        typer.secho(f"Error processing image: {ve}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    finally:
        db.close()


if __name__ == "__main__":
    app()
