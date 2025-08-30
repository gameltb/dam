import logging
from typing import Optional

from ..commands import IngestWebAssetCommand
from dam.core.systems import handles_command
from dam.core.transaction import EcsTransaction
from dam.models.core.entity import Entity
from dam_source.models.source_info import source_types
from dam_source.models.source_info.original_source_info_component import OriginalSourceInfoComponent
from dam_source.models.source_info.web_source_component import WebSourceComponent
from dam_source.models.source_info.website_profile_component import WebsiteProfileComponent

logger = logging.getLogger(__name__)


@handles_command(IngestWebAssetCommand)
async def handle_ingest_web_asset_command(
    cmd: IngestWebAssetCommand,
    transaction: EcsTransaction,
):
    """
    Handles the command to ingest an asset from a web source.
    Primarily stores metadata and URLs. File download/processing is deferred.
    """
    logger.info(
        f"System handling IngestWebAssetCommand for URL: {cmd.source_url} from website: {cmd.website_identifier_url} in world {cmd.world_name}"
    )

    # 1. Find or Create Website Entity
    website_entity: Optional[Entity] = None
    existing_website_profiles = await transaction.find_entities_by_component_attribute_value(
        WebsiteProfileComponent, "main_url", cmd.website_identifier_url
    )
    if existing_website_profiles:
        website_entity = existing_website_profiles[0]
        logger.info(f"Found existing Website Entity ID {website_entity.id} for URL {cmd.website_identifier_url}")
    else:
        website_entity = await transaction.create_entity()
        await transaction.flush()
        logger.info(f"Creating new Website Entity ID {website_entity.id} for URL {cmd.website_identifier_url}")

        website_name = cmd.metadata_payload.get("website_name") if cmd.metadata_payload else None
        if not website_name:
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(cmd.website_identifier_url)
                website_name = parsed_url.netloc.replace("www.", "")
            except Exception:
                website_name = "Unknown Website"

        profile_comp = WebsiteProfileComponent(
            name=website_name,
            main_url=cmd.website_identifier_url,
            description=cmd.metadata_payload.get("website_description") if cmd.metadata_payload else None,
        )
        await transaction.add_component_to_entity(website_entity.id, profile_comp)

    # 2. Create Asset Entity
    asset_entity = await transaction.create_entity()
    await transaction.flush()
    logger.info(f"Creating new Asset Entity ID {asset_entity.id} for web asset from URL: {cmd.source_url}")

    # 3. Create OriginalSourceInfoComponent for the Asset Entity
    original_filename = cmd.metadata_payload.get("asset_title") if cmd.metadata_payload else None
    if not original_filename:
        try:
            original_filename = cmd.source_url.split("/")[-1] or f"web_asset_{asset_entity.id}"
        except Exception:
            original_filename = f"web_asset_{asset_entity.id}"

    osi_comp = OriginalSourceInfoComponent(
        source_type=source_types.SOURCE_TYPE_WEB_SOURCE,
    )
    await transaction.add_component_to_entity(asset_entity.id, osi_comp)

    # 4. Create WebSourceComponent for the Asset Entity
    web_source_data = {
        "website_entity_id": website_entity.id,
        "source_url": cmd.source_url,
        "original_file_url": cmd.original_file_url,
    }
    if cmd.metadata_payload:
        web_source_data["gallery_id"] = cmd.metadata_payload.get("gallery_id")
        web_source_data["uploader_name"] = cmd.metadata_payload.get("uploader_name")
        web_source_data["uploader_url"] = cmd.metadata_payload.get("uploader_url")
        web_source_data["asset_title"] = cmd.metadata_payload.get("asset_title", original_filename)
        web_source_data["asset_description"] = cmd.metadata_payload.get("asset_description")

        upload_date_str = cmd.metadata_payload.get("upload_date")
        if upload_date_str:
            try:
                from datetime import datetime
                web_source_data["upload_date"] = datetime.fromisoformat(upload_date_str.replace("Z", "+00:00"))
            except ValueError:
                logger.warning(f"Could not parse upload_date string '{upload_date_str}' for {cmd.source_url}")
                web_source_data["upload_date"] = None
        else:
            web_source_data["upload_date"] = None

        web_source_data["raw_metadata_dump"] = cmd.metadata_payload

    if cmd.tags:
        import json
        web_source_data["tags_json"] = json.dumps(cmd.tags)

    valid_web_source_fields = {
        "website_entity_id",
        "source_url",
        "original_file_url",
        "gallery_id",
        "uploader_name",
        "uploader_url",
        "upload_date",
        "asset_title",
        "asset_description",
        "tags_json",
        "raw_metadata_dump",
    }
    cleaned_web_source_data = {
        k: v for k, v in web_source_data.items() if k in valid_web_source_fields and v is not None
    }

    web_comp = WebSourceComponent(**cleaned_web_source_data)
    await transaction.add_component_to_entity(asset_entity.id, web_comp)

    logger.info(
        f"Finished IngestWebAssetCommand for Asset Entity ID {asset_entity.id} (Website Entity ID {website_entity.id}) from URL: {cmd.source_url}"
    )


__all__ = ["handle_ingest_web_asset_command"]
