from pathlib import Path

import typer
from sqlalchemy import select  # Added for SQLAlchemy 2.0 queries
from typing_extensions import Annotated

from dam.core.database import SessionLocal, create_db_and_tables
from dam.core.logging_config import setup_logging  # Import the setup function
from dam.models import (
    ContentHashComponent,
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


@app.command(name="query-by-hash")
def cli_query_by_hash(
    hash_value: Annotated[str, typer.Argument(..., help="The content hash value to search for.")],
    hash_type: Annotated[str, typer.Option(help="Type of the hash (e.g., sha256).")] = "sha256",
):
    """
    Finds and displays information about an asset entity by its content hash.
    """
    typer.echo(f"Querying for asset with {hash_type} hash: {hash_value}")
    db = SessionLocal()
    try:
        entity = asset_service.find_entity_by_content_hash(db, hash_value, hash_type)
        if entity:
            typer.secho(f"Found Entity ID: {entity.id}", fg=typer.colors.CYAN)

            # Fetch and display components using SQLAlchemy 2.0 style
            chc_stmt = select(ContentHashComponent).where(ContentHashComponent.entity_id == entity.id)
            for chc_component in db.execute(chc_stmt).scalars().all():
                typer.echo(f"  Content Hash: Type='{chc_component.hash_type}', Value='{chc_component.hash_value}'")

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


if __name__ == "__main__":
    app()
