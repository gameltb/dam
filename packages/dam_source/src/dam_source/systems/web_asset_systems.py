"""Defines systems for handling web assets."""

import json
import logging
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.models.core.entity import Entity

from ..commands import IngestWebAssetCommand
from ..models.source_info.web_source_component import WebSourceComponent
from ..models.source_info.website_profile_component import WebsiteProfileComponent

logger = logging.getLogger(__name__)


async def _find_or_create_website_entity(transaction: WorldTransaction, cmd: IngestWebAssetCommand) -> Entity:
    """Find an existing website entity or create a new one."""
    existing_profiles = await transaction.find_entities_by_component_attribute_value(
        WebsiteProfileComponent, "main_url", cmd.website_identifier_url
    )
    if existing_profiles:
        website_entity = existing_profiles[0]
        logger.info(
            "Found existing Website Entity ID %s for URL %s",
            website_entity.id,
            cmd.website_identifier_url,
        )
        return website_entity

    website_entity = await transaction.create_entity()
    await transaction.flush()
    logger.info(
        "Creating new Website Entity ID %s for URL %s",
        website_entity.id,
        cmd.website_identifier_url,
    )

    website_name = cmd.metadata_payload.get("website_name") if cmd.metadata_payload else None
    if not website_name:
        try:
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
    return website_entity


def _prepare_web_source_data(
    cmd: IngestWebAssetCommand, website_entity: Entity, asset_entity: Entity
) -> dict[str, Any]:
    """Prepare the data for the WebSourceComponent."""
    original_filename = cmd.metadata_payload.get("asset_title") if cmd.metadata_payload else None
    if not original_filename:
        try:
            original_filename = cmd.source_url.split("/")[-1] or f"web_asset_{asset_entity.id}"
        except Exception:
            original_filename = f"web_asset_{asset_entity.id}"

    data: dict[str, Any] = {
        "website_entity_id": website_entity.id,
        "source_url": cmd.source_url,
        "original_file_url": cmd.original_file_url,
    }

    if cmd.metadata_payload:
        data.update(
            {
                "gallery_id": cmd.metadata_payload.get("gallery_id"),
                "uploader_name": cmd.metadata_payload.get("uploader_name"),
                "uploader_url": cmd.metadata_payload.get("uploader_url"),
                "asset_title": cmd.metadata_payload.get("asset_title", original_filename),
                "asset_description": cmd.metadata_payload.get("asset_description"),
                "raw_metadata_dump": cmd.metadata_payload,
            }
        )
        upload_date_str = cmd.metadata_payload.get("upload_date")
        if upload_date_str:
            try:
                data["upload_date"] = datetime.fromisoformat(upload_date_str.replace("Z", "+00:00"))
            except ValueError:
                logger.warning(
                    "Could not parse upload_date string '%s' for %s",
                    upload_date_str,
                    cmd.source_url,
                )
                data["upload_date"] = None
        else:
            data["upload_date"] = None

    if cmd.tags:
        data["tags_json"] = json.dumps(cmd.tags)

    return data


@system(on_command=IngestWebAssetCommand)
async def handle_ingest_web_asset_command(
    cmd: IngestWebAssetCommand,
    transaction: WorldTransaction,
) -> None:
    """
    Handle the command to ingest an asset from a web source.

    This system primarily stores metadata and URLs. File download and
    processing are deferred.
    """
    logger.info(
        "System handling IngestWebAssetCommand for URL: %s from website: %s in world %s",
        cmd.source_url,
        cmd.website_identifier_url,
        cmd.world_name,
    )

    website_entity = await _find_or_create_website_entity(transaction, cmd)
    asset_entity = await transaction.create_entity()
    await transaction.flush()
    logger.info(
        "Creating new Asset Entity ID %s for web asset from URL: %s",
        asset_entity.id,
        cmd.source_url,
    )

    web_source_data = _prepare_web_source_data(cmd, website_entity, asset_entity)
    valid_fields = {f.name for f in WebSourceComponent.__table__.columns}
    cleaned_data = {k: v for k, v in web_source_data.items() if k in valid_fields and v is not None}

    if "website_entity_id" in cleaned_data and "source_url" in cleaned_data:
        web_comp = WebSourceComponent(**cleaned_data)
        await transaction.add_component_to_entity(asset_entity.id, web_comp)

    logger.info(
        "Finished IngestWebAssetCommand for Asset Entity ID %s (Website Entity ID %s) from URL: %s",
        asset_entity.id,
        website_entity.id,
        cmd.source_url,
    )


__all__ = ["handle_ingest_web_asset_command"]
