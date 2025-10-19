"""Defines the systems for handling image-specific metadata and commands."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any

import aiofiles
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.utils.hash_utils import HashAlgorithm, calculate_hashes_from_stream
from dam_fs.functions import file_operations
from dam_fs.models.filename_component import FilenameComponent
from PIL import Image
from sqlalchemy import select as sql_select

from dam_media_image.commands import FindSimilarImagesCommand
from dam_media_image.events import ImageAssetDetected
from dam_media_image.functions import image_hashing_functions as image_hashing_service
from dam_media_image.types import SimilarityResult
from dam_media_image.models.hashes.base_image_perceptual_hash_component import (
    BaseImagePerceptualHashComponent,
)
from dam_media_image.models.hashes.image_perceptual_hash_ahash_component import (
    ImagePerceptualAHashComponent,
)
from dam_media_image.models.hashes.image_perceptual_hash_dhash_component import (
    ImagePerceptualDHashComponent,
)
from dam_media_image.models.hashes.image_perceptual_hash_phash_component import (
    ImagePerceptualPHashComponent,
)
from dam_media_image.models.properties.image_dimensions_component import (
    ImageDimensionsComponent,
)

if TYPE_CHECKING:
    import imagehash
    from imagehash import ImageHash

try:
    import imagehash
except ImportError:
    imagehash = None  # type: ignore

logger = logging.getLogger(__name__)


async def _find_source_entity_id(transaction: WorldTransaction, cmd: FindSimilarImagesCommand) -> int | None:
    """Find the entity ID of the source image to exclude it from results."""
    try:
        async with aiofiles.open(cmd.image_path, "rb") as f:
            content = await f.read()
        stream = io.BytesIO(content)
        hashes = calculate_hashes_from_stream(stream, {HashAlgorithm.SHA256})
        source_content_hash_bytes = hashes[HashAlgorithm.SHA256]
        source_entity = await transaction.find_entity_by_content_hash(source_content_hash_bytes, "sha256")  # type: ignore
        return source_entity.id if source_entity else None
    except Exception as e:
        logger.warning("Could not determine source entity for %s to exclude from results: %s", cmd.image_path.name, e)
        return None


async def _find_similar_hashes(
    transaction: WorldTransaction,
    source_entity_id: int | None,
    input_hash_obj: ImageHash,
    hash_component_class: type[BaseImagePerceptualHashComponent],
    threshold: int,
    hash_type: str,
) -> list[dict[str, Any]]:
    """Find similar images for a given hash type."""
    matches: list[dict[str, Any]] = []
    stmt = sql_select(hash_component_class)
    result = await transaction.session.execute(stmt)
    db_hash_components = result.scalars().all()

    for comp in db_hash_components:
        if source_entity_id and comp.entity_id == source_entity_id:
            continue
        try:
            assert imagehash is not None
            db_hash_hex = comp.hash_value.hex()
            db_hash_obj = imagehash.hex_to_hash(db_hash_hex)
            distance = input_hash_obj - db_hash_obj
            if distance <= threshold:
                entity = await transaction.get_entity(comp.entity_id)
                if entity:
                    fnc = await transaction.get_component(entity.id, FilenameComponent)
                    matches.append(
                        {
                            "entity_id": entity.id,
                            "original_filename": fnc.filename if fnc else "N/A",
                            "match_type": f"{hash_type}_match",
                            "distance": distance,
                            "hash_type": hash_type,
                        }
                    )
        except Exception:
            logger.exception("Error comparing %s for entity %d", hash_type, comp.entity_id)
    return matches


def _consolidate_matches(potential_matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Consolidate matches from different hash types, keeping the best one."""
    final_matches_map: dict[int, dict[str, Any]] = {}
    for match in potential_matches:
        entity_id = match["entity_id"]
        if isinstance(entity_id, int) and (
            entity_id not in final_matches_map
            or (
                match["distance"] is not None and int(match["distance"]) < int(final_matches_map[entity_id]["distance"])
            )
        ):
            final_matches_map[entity_id] = match
    similar_entities_info = list(final_matches_map.values())
    similar_entities_info.sort(key=lambda x: (x["distance"], x["entity_id"]))
    return similar_entities_info


@system(on_command=FindSimilarImagesCommand)
async def handle_find_similar_images_command(
    cmd: FindSimilarImagesCommand,
    transaction: WorldTransaction,
) -> SimilarityResult:
    """Handle the command to find similar images based on perceptual hashes."""
    logger.info("System handling FindSimilarImagesCommand for image: %s (Req ID: %s)", cmd.image_path, cmd.request_id)
    try:
        if not imagehash:
            msg = "ImageHash library not available. Cannot perform similarity search."
            logger.warning("[QueryResult RequestID: %s] %s", cmd.request_id, msg)
            return [{"error": msg}]

        input_hashes = await image_hashing_service.generate_perceptual_hashes_async(cmd.image_path)
        if not input_hashes:
            msg = f"Could not generate perceptual hashes for {cmd.image_path.name}."
            logger.warning("[QueryResult RequestID: %s] %s", cmd.request_id, msg)
            return [{"error": msg}]

        source_entity_id = await _find_source_entity_id(transaction, cmd)
        potential_matches: list[dict[str, Any]] = []

        if "phash" in input_hashes:
            input_phash_obj = imagehash.hex_to_hash(input_hashes["phash"])
            potential_matches.extend(
                await _find_similar_hashes(
                    transaction,
                    source_entity_id,
                    input_phash_obj,
                    ImagePerceptualPHashComponent,
                    cmd.phash_threshold,
                    "phash",
                )
            )
        if "ahash" in input_hashes:
            input_ahash_obj = imagehash.hex_to_hash(input_hashes["ahash"])
            potential_matches.extend(
                await _find_similar_hashes(
                    transaction,
                    source_entity_id,
                    input_ahash_obj,
                    ImagePerceptualAHashComponent,
                    cmd.ahash_threshold,
                    "ahash",
                )
            )
        if "dhash" in input_hashes:
            input_dhash_obj = imagehash.hex_to_hash(input_hashes["dhash"])
            potential_matches.extend(
                await _find_similar_hashes(
                    transaction,
                    source_entity_id,
                    input_dhash_obj,
                    ImagePerceptualDHashComponent,
                    cmd.dhash_threshold,
                    "dhash",
                )
            )

        similar_entities_info = _consolidate_matches(potential_matches)
        logger.info("[QueryResult RequestID: %s] Found %d similar images.", cmd.request_id, len(similar_entities_info))
        return similar_entities_info

    except ValueError as ve:
        logger.warning("[QueryResult RequestID: %s] Error processing image for similarity: %s", cmd.request_id, ve)
        return [{"error": str(ve)}]
    except Exception as e:
        logger.exception("[QueryResult RequestID: %s] Unexpected error in similarity search: %s", cmd.request_id, e)
        raise


@system(on_event=ImageAssetDetected)
async def process_image_metadata_system(
    event: ImageAssetDetected,
    transaction: WorldTransaction,
    world: World,
) -> None:
    """Listen for an image asset being detected and extract its metadata."""
    logger.info("Processing image metadata for entity %d", event.entity.id)
    try:
        existing_component = await transaction.get_component(event.entity.id, ImageDimensionsComponent)
        if existing_component:
            logger.info("Entity %d already has ImageDimensionsComponent. Skipping.", event.entity.id)
            return

        file_path = await file_operations.get_file_path_by_id(world, transaction, event.file_id)
        if not file_path:
            logger.warning(
                "Could not find file path for file_id %d on entity %d. Cannot process image metadata.",
                event.file_id,
                event.entity.id,
            )
            return

        with Image.open(file_path) as img:
            width, height = img.size

        dimensions_component = ImageDimensionsComponent()
        dimensions_component.width_pixels = width
        dimensions_component.height_pixels = height
        await transaction.add_component_to_entity(event.entity.id, dimensions_component)

        logger.info("Successfully added ImageDimensionsComponent to entity %d.", event.entity.id)

    except Exception as e:
        logger.exception("Failed during image metadata processing for entity %d: %s", event.entity.id, e)
        raise
