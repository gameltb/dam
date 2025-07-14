"""
Example script to ingest assets downloaded by gallery-dl into the DAM.
This script is intended to be called by gallery-dl's "exec" post-processor.

Example gallery-dl config:
{
  "extractor": {
    // other gallery-dl settings
    "postprocessors": [
      {
        "name": "exec",
        "command": [
          "python",
          "/path/to/dam/examples/ingest_from_gallery_dl.py",
          "--world", "my_dam_world",
          "--source-url", "{url}",
          "--downloaded-filepath", "{_path}",
          "--metadata-filepath", "{_metadata_path}"
        ]
      }
    ]
  }
}

Make sure to run `gallery-dl --write-metadata ...` for {_metadata_path} to be available.
"""

import asyncio
import binascii
import json
import logging
from pathlib import Path

import typer
from dam.core import config as app_config
from dam.core.logging_config import setup_logging
from dam.core.world import World, create_and_register_all_worlds_from_settings, get_world
from dam.core.world_setup import register_core_systems
from dam.models import (
    ContentHashMD5Component,
    ContentHashSHA256Component,
    Entity,
    FileLocationComponent,
    FilePropertiesComponent,
    NeedsMetadataExtractionComponent,
    OriginalSourceInfoComponent,
    WebsiteProfileComponent,  # Needed if creating website entities
    WebSourceComponent,
)
from dam.models.source_info import source_types
from dam.resources.file_storage_resource import FileStorageResource
from dam.services import ecs_service, file_operations
from typing_extensions import Annotated

logger = logging.getLogger(__name__)
cli_app = typer.Typer()


def get_website_entity(session, website_identifier_url: str, metadata_payload: dict) -> Entity:
    """Finds or creates a WebsiteProfile entity."""
    existing_profiles = ecs_service.find_entities_by_component_attribute_value(session, WebsiteProfileComponent, "main_url", website_identifier_url)
    if existing_profiles:
        logger.info(f"Found existing Website Entity ID {existing_profiles[0].id} for URL {website_identifier_url}")
        return existing_profiles[0]

    website_entity = ecs_service.create_entity(session)
    logger.info(f"Creating new Website Entity ID {website_entity.id} for URL {website_identifier_url}")

    website_name = metadata_payload.get("website_name")
    if not website_name:
        try:
            from urllib.parse import urlparse

            parsed_url = urlparse(website_identifier_url)
            website_name = parsed_url.netloc.replace("www.", "")
        except Exception:
            website_name = "Unknown Website"

    profile_comp = WebsiteProfileComponent(
        entity_id=website_entity.id,
        entity=website_entity,
        name=website_name,
        main_url=website_identifier_url,
        description=metadata_payload.get("website_description"),
    )
    ecs_service.add_component_to_entity(session, website_entity.id, profile_comp)
    return website_entity


async def ingest_gallery_dl_asset_async(
    world: World,
    source_url: str,
    downloaded_filepath: Path,
    metadata_filepath: Path,
):
    """
    Asynchronous core ingestion logic.
    """
    logger.info(f"Ingesting '{downloaded_filepath.name}' from source URL '{source_url}' into world '{world.name}'")

    metadata_payload = {}
    if metadata_filepath.exists():
        try:
            with open(metadata_filepath, "r", encoding="utf-8") as f:
                metadata_payload = json.load(f)
            logger.debug(f"Loaded metadata from {metadata_filepath}")
        except Exception as e:
            logger.warning(f"Could not load metadata from {metadata_filepath}: {e}")

    # 1. Get file properties from the downloaded file
    try:
        actual_filename, size_bytes, mime_type = file_operations.get_file_properties(downloaded_filepath)
        file_content = await file_operations.read_file_async(downloaded_filepath)
    except Exception as e:
        logger.error(f"Error reading file properties or content for {downloaded_filepath}: {e}")
        return

    # 2. Store file in DAM using FileStorageResource
    # This assumes FileStorageResource is registered and accessible on the world
    try:
        file_storage_resource = world.get_resource(FileStorageResource)
        if not file_storage_resource:
            logger.error("FileStorageResource not found in world. Cannot store file.")
            return
    except Exception as e:
        logger.error(f"Error getting FileStorageResource: {e}")
        return

    content_hash_sha256_hex, physical_storage_path_suffix = file_storage_resource.store_file(file_content, original_filename=actual_filename)
    content_hash_sha256_bytes = binascii.unhexlify(content_hash_sha256_hex)

    async with world.db_manager.get_session() as session:
        # 3. Check for existing entity by SHA256 hash
        existing_entity = ecs_service.find_entity_by_content_hash(session, content_hash_sha256_bytes, "sha256")
        entity: Entity
        created_new_entity = False

        if existing_entity:
            entity = existing_entity
            logger.info(f"Content SHA256 {content_hash_sha256_hex[:12]} for '{actual_filename}' already exists as Entity ID {entity.id}.")
        else:
            created_new_entity = True
            entity = ecs_service.create_entity(session)
            logger.info(f"Creating new Entity ID {entity.id} for '{actual_filename}' (SHA256: {content_hash_sha256_hex[:12]}).")

            # Add SHA256 hash component
            chc_sha256 = ContentHashSHA256Component(entity=entity, hash_value=content_hash_sha256_bytes)
            ecs_service.add_component_to_entity(session, entity.id, chc_sha256)

            # Add MD5 hash component
            md5_hash_hex = await file_operations.calculate_md5_async(downloaded_filepath)
            md5_hash_bytes = binascii.unhexlify(md5_hash_hex)
            chc_md5 = ContentHashMD5Component(entity=entity, hash_value=md5_hash_bytes)
            ecs_service.add_component_to_entity(session, entity.id, chc_md5)

            # Add FilePropertiesComponent
            fpc = FilePropertiesComponent(
                entity=entity,
                original_filename=actual_filename,  # Name of the downloaded file
                file_size_bytes=size_bytes,
                mime_type=mime_type,
            )
            ecs_service.add_component_to_entity(session, entity.id, fpc)

            # Add FileLocationComponent (DAM managed storage)
            flc = FileLocationComponent(
                entity=entity,
                content_identifier=content_hash_sha256_hex,
                storage_type="dam_managed_storage",  # Or "local_cas" if that's your convention
                physical_path_or_key=str(physical_storage_path_suffix),
                contextual_filename=actual_filename,
            )
            ecs_service.add_component_to_entity(session, entity.id, flc)

        # 4. Add OriginalSourceInfoComponent (classifying it as web source)
        # Check if this specific source URL is already linked to this entity
        existing_osis = ecs_service.get_components(session, entity.id, OriginalSourceInfoComponent)
        source_already_linked = False
        for osi in existing_osis:
            if osi.source_type == source_types.SOURCE_TYPE_WEB_SOURCE:
                # Further check if a WebSourceComponent with this source_url exists for this osi
                # This check needs to be more robust if multiple web sources can lead to same entity
                wscs = ecs_service.get_components(session, entity.id, WebSourceComponent)
                if any(w.source_url == source_url for w in wscs):
                    source_already_linked = True
                    logger.info(f"Entity {entity.id} already has OriginalSourceInfo for web source URL {source_url}.")
                    break

        if not source_already_linked:
            osi_comp = OriginalSourceInfoComponent(
                entity=entity,
                source_type=source_types.SOURCE_TYPE_WEB_SOURCE,
            )
            ecs_service.add_component_to_entity(session, entity.id, osi_comp)

        # 5. Add WebSourceComponent
        # Determine website_identifier_url from metadata or source_url
        # gallery-dl metadata usually has 'parent' or similar for the gallery page URL
        # and 'website' for the site name (e.g. deviantart, twitter)
        # For simplicity, we might use the source_url's domain as website_identifier_url if not explicitly provided
        website_identifier_url = metadata_payload.get("parent")  # Prefer gallery page if available
        if not website_identifier_url:
            try:
                from urllib.parse import urlparse

                parsed_source_url = urlparse(source_url)
                website_identifier_url = f"{parsed_source_url.scheme}://{parsed_source_url.netloc}"
            except Exception:
                logger.warning(f"Could not derive website identifier URL from {source_url}")
                website_identifier_url = source_url  # Fallback

        website_entity = get_website_entity(session, website_identifier_url, metadata_payload)

        # Check if this WebSourceComponent already exists for the entity
        existing_wscs = ecs_service.get_components(session, entity.id, WebSourceComponent)
        wsc_exists = any(w.source_url == source_url and w.website_entity_id == website_entity.id for w in existing_wscs)

        if not wsc_exists:
            web_source_data = {
                "entity_id": entity.id,
                "entity": entity,
                "website_entity_id": website_entity.id,
                "source_url": source_url,  # The specific page URL of the asset
                "original_file_url": metadata_payload.get("file_url", source_url),  # Direct file URL if different
                "gallery_id": metadata_payload.get(metadata_payload.get("id_key", "id")),  # e.g., post ID
                "uploader_name": metadata_payload.get("uploader") or metadata_payload.get("artist"),
                "uploader_url": metadata_payload.get("uploader_url"),
                "asset_title": metadata_payload.get("title"),
                "asset_description": metadata_payload.get("description") or metadata_payload.get("caption"),
                "raw_metadata_dump": metadata_payload,
            }
            upload_date_str = metadata_payload.get("date")  # gallery-dl often uses 'date'
            if upload_date_str:
                try:
                    from datetime import datetime, timezone

                    # gallery-dl date format is often like "YYYYMMDD" or "YYYY-MM-DD HH:MM:SS"
                    # Handle various potential formats or make it robust
                    if len(upload_date_str) == 8 and upload_date_str.isdigit():  # YYYYMMDD
                        dt_obj = datetime.strptime(upload_date_str, "%Y%m%d")
                    elif isinstance(upload_date_str, (int, float)):  # Timestamp
                        dt_obj = datetime.fromtimestamp(upload_date_str, tz=timezone.utc)
                    else:  # Try ISO format or with spaces
                        dt_obj = datetime.fromisoformat(upload_date_str.replace(" ", "T").replace("Z", "+00:00"))
                    web_source_data["upload_date"] = dt_obj.astimezone(timezone.utc) if dt_obj.tzinfo is None else dt_obj

                except ValueError as e:
                    logger.warning(f"Could not parse upload_date string '{upload_date_str}' for {source_url}: {e}")

            tags_list = metadata_payload.get("tags", [])
            if isinstance(tags_list, str):  # Sometimes tags are space-separated string
                tags_list = tags_list.split()
            if tags_list:
                web_source_data["tags_json"] = json.dumps(list(set(tags_list)))  # Ensure unique tags

            web_comp = WebSourceComponent(**{k: v for k, v in web_source_data.items() if hasattr(WebSourceComponent, k) and v is not None})
            ecs_service.add_component_to_entity(session, entity.id, web_comp)
            logger.info(f"Added WebSourceComponent for Entity ID {entity.id} from URL {source_url}")
        else:
            logger.info(f"WebSourceComponent for Entity ID {entity.id} and URL {source_url} already exists.")

        # 6. Add NeedsMetadataExtractionComponent if new entity or if properties were missing
        if created_new_entity or not ecs_service.get_components(session, entity.id, FilePropertiesComponent):  # Re-check FPC
            if not ecs_service.get_components(session, entity.id, NeedsMetadataExtractionComponent):
                marker_comp = NeedsMetadataExtractionComponent(entity=entity)
                ecs_service.add_component_to_entity(session, entity.id, marker_comp)
                logger.info(f"Marked Entity ID {entity.id} for metadata extraction.")

        await session.commit()
        logger.info(f"Successfully processed and committed Entity ID {entity.id} for {actual_filename}")


@cli_app.command()
def main(
    world_name: Annotated[str, typer.Option(help="Name of the ECS world to operate on.", envvar="DAM_CURRENT_WORLD")],
    source_url: Annotated[str, typer.Option(help="Original URL of the asset page.")],
    downloaded_filepath_str: Annotated[
        str,
        typer.Option("--downloaded-filepath", help="Path to the downloaded asset file.", exists=True, resolve_path=True),
    ],
    metadata_filepath_str: Annotated[
        str,
        typer.Option("--metadata-filepath", help="Path to the gallery-dl metadata JSON file.", exists=True, resolve_path=True),
    ],
):
    """
    Ingest an asset downloaded by gallery-dl into the DAM.
    """
    setup_logging()
    downloaded_filepath = Path(downloaded_filepath_str)
    metadata_filepath = Path(metadata_filepath_str)

    # Initialize worlds
    try:
        create_and_register_all_worlds_from_settings(app_settings=app_config.settings)
    except Exception as e:
        logger.critical(f"Could not initialize worlds from settings: {e}", exc_info=True)
        raise typer.Exit(code=1)

    # Register core systems (important for resources like FileStorageResource)

    for w_instance in get_world(None, get_all=True):  # type: ignore
        register_core_systems(w_instance)

    target_world = get_world(world_name)
    if not target_world:
        logger.error(f"World '{world_name}' not found or not initialized correctly.")
        raise typer.Exit(code=1)

    logger.info(f"Operating on world: '{target_world.name}'")

    asyncio.run(
        ingest_gallery_dl_asset_async(
            world=target_world,
            source_url=source_url,
            downloaded_filepath=downloaded_filepath,
            metadata_filepath=metadata_filepath,
        )
    )
    logger.info("Ingestion process finished.")


if __name__ == "__main__":
    # Create dam/examples directory if it doesn't exist
    # This is mainly for the sandbox environment when creating the file.
    # In a real deployment, the directory structure would exist.
    current_script_path = Path(__file__)
    examples_dir = current_script_path.parent
    if not examples_dir.exists():
        try:
            examples_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # In the sandbox, this might fail if parent dirs are restricted
            # but it's fine as long as the tool creates the file in the right relative path.
            print(f"Could not create examples directory (expected in sandbox): {e}")
            pass

    cli_app()

# Ensure FileStorageResource is registered to the world during its setup for this script to work.
# This typically happens in world_setup.py or similar central registration logic.
# The `register_core_systems` call in `main` should handle this if FileStorageResource
# is part of the core resources.

# Also, ensure `dam.models.source_info.source_types` is accessible.
# Add `from dam.models.source_info import source_types` to where it's used.
# This was done in the script.
