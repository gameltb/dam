import io
import logging
from typing import Any

import aiofiles
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.utils.hash_utils import HashAlgorithm, calculate_hashes_from_stream
from dam_fs.functions import file_operations
from dam_fs.models.filename_component import FilenameComponent
from PIL import Image
from sqlalchemy import select as sql_select

from ..commands import FindSimilarImagesCommand
from ..events import ImageAssetDetected
from ..functions import image_hashing_functions as image_hashing_service
from ..models.hashes.image_perceptual_hash_ahash_component import (
    ImagePerceptualAHashComponent,
)
from ..models.hashes.image_perceptual_hash_dhash_component import (
    ImagePerceptualDHashComponent,
)
from ..models.hashes.image_perceptual_hash_phash_component import (
    ImagePerceptualPHashComponent,
)
from ..models.properties.image_dimensions_component import ImageDimensionsComponent

try:
    import imagehash
except ImportError:
    imagehash = None  # type: ignore

logger = logging.getLogger(__name__)


@system(on_command=FindSimilarImagesCommand)
async def handle_find_similar_images_command(
    cmd: FindSimilarImagesCommand,
    transaction: WorldTransaction,
) -> list[dict[str, Any]] | None:
    logger.info(f"System handling FindSimilarImagesCommand for image: {cmd.image_path} (Req ID: {cmd.request_id})")

    try:
        if not imagehash:
            msg = "ImageHash library not available. Cannot perform similarity search."
            logger.warning(f"[QueryResult RequestID: {cmd.request_id}] {msg}")
            return [{"error": msg}]

        input_hashes = await image_hashing_service.generate_perceptual_hashes_async(cmd.image_path)
        if not input_hashes:
            msg = f"Could not generate perceptual hashes for {cmd.image_path.name}."
            logger.warning(f"[QueryResult RequestID: {cmd.request_id}] {msg}")
            return [{"error": msg}]

        input_phash_obj = imagehash.hex_to_hash(input_hashes["phash"]) if "phash" in input_hashes else None
        input_ahash_obj = imagehash.hex_to_hash(input_hashes["ahash"]) if "ahash" in input_hashes else None
        input_dhash_obj = imagehash.hex_to_hash(input_hashes["dhash"]) if "dhash" in input_hashes else None

        source_entity_id = None
        try:
            async with aiofiles.open(cmd.image_path, "rb") as f:
                content = await f.read()
            stream = io.BytesIO(content)
            hashes = calculate_hashes_from_stream(stream, {HashAlgorithm.SHA256})
            source_content_hash_bytes = hashes[HashAlgorithm.SHA256]
            source_entity = await transaction.find_entity_by_content_hash(source_content_hash_bytes, "sha256")  # type: ignore
            if source_entity:
                source_entity_id = source_entity.id
        except Exception as e_src:
            logger.warning(
                f"Could not determine source entity for {cmd.image_path.name} to exclude from results: {e_src}"
            )

        potential_matches: list[dict[str, Any]] = []

        if input_phash_obj:
            all_phashes_stmt = sql_select(ImagePerceptualPHashComponent)
            result_phashes = await transaction.session.execute(all_phashes_stmt)
            db_phashes_components = result_phashes.scalars().all()
            for p_comp in db_phashes_components:
                if source_entity_id and p_comp.entity_id == source_entity_id:
                    continue
                try:
                    db_phash_hex = p_comp.hash_value.hex()
                    db_phash_obj = imagehash.hex_to_hash(db_phash_hex)
                    distance = input_phash_obj - db_phash_obj
                    if distance <= cmd.phash_threshold:
                        entity = await transaction.get_entity(p_comp.entity_id)
                        if entity:
                            fnc = await transaction.get_component(entity.id, FilenameComponent)
                            potential_matches.append(
                                {
                                    "entity_id": entity.id,
                                    "original_filename": fnc.filename if fnc else "N/A",
                                    "match_type": "phash_match",
                                    "distance": distance,
                                    "hash_type": "phash",
                                }
                            )
                except Exception as e_cmp:
                    logger.warning(f"Error comparing pHash for entity {p_comp.entity_id}: {e_cmp}")

        if input_ahash_obj:
            all_ahashes_stmt = sql_select(ImagePerceptualAHashComponent)
            result_ahashes = await transaction.session.execute(all_ahashes_stmt)
            db_ahashes_components = result_ahashes.scalars().all()
            for a_comp in db_ahashes_components:
                if source_entity_id and a_comp.entity_id == source_entity_id:
                    continue
                try:
                    db_ahash_hex = a_comp.hash_value.hex()
                    db_ahash_obj = imagehash.hex_to_hash(db_ahash_hex)
                    distance = input_ahash_obj - db_ahash_obj
                    if distance <= cmd.ahash_threshold:
                        entity = await transaction.get_entity(a_comp.entity_id)
                        if entity:
                            fnc = await transaction.get_component(entity.id, FilenameComponent)
                            potential_matches.append(
                                {
                                    "entity_id": entity.id,
                                    "original_filename": fnc.filename if fnc else "N/A",
                                    "match_type": "ahash_match",
                                    "distance": distance,
                                    "hash_type": "ahash",
                                }
                            )
                except Exception as e_cmp:
                    logger.warning(f"Error comparing aHash for entity {a_comp.entity_id}: {e_cmp}")

        if input_dhash_obj:
            all_dhashes_stmt = sql_select(ImagePerceptualDHashComponent)
            result_dhashes = await transaction.session.execute(all_dhashes_stmt)
            db_dhashes_components = result_dhashes.scalars().all()
            for d_comp in db_dhashes_components:
                if source_entity_id and d_comp.entity_id == source_entity_id:
                    continue
                try:
                    db_dhash_hex = d_comp.hash_value.hex()
                    db_dhash_obj = imagehash.hex_to_hash(db_dhash_hex)
                    distance = input_dhash_obj - db_dhash_obj
                    if distance <= cmd.dhash_threshold:
                        entity = await transaction.get_entity(d_comp.entity_id)
                        if entity:
                            fnc = await transaction.get_component(entity.id, FilenameComponent)
                            potential_matches.append(
                                {
                                    "entity_id": entity.id,
                                    "original_filename": fnc.filename if fnc else "N/A",
                                    "match_type": "dhash_match",
                                    "distance": distance,
                                    "hash_type": "dhash",
                                }
                            )
                except Exception as e_cmp:
                    logger.warning(f"Error comparing dHash for entity {d_comp.entity_id}: {e_cmp}")

        final_matches_map: dict[int, dict[str, Any]] = {}
        for match in potential_matches:
            entity_id = match["entity_id"]
            if isinstance(entity_id, int) and (
                entity_id not in final_matches_map
                or (
                    match["distance"] is not None
                    and int(match["distance"]) < int(final_matches_map[entity_id]["distance"])
                )
            ):
                final_matches_map[entity_id] = match

        similar_entities_info = list(final_matches_map.values())
        similar_entities_info.sort(key=lambda x: (x["distance"], x["entity_id"]))

        logger.info(f"[QueryResult RequestID: {cmd.request_id}] Found {len(similar_entities_info)} similar images.")
        return similar_entities_info

    except ValueError as ve:
        logger.warning(f"[QueryResult RequestID: {cmd.request_id}] Error processing image for similarity: {ve}")
        return [{"error": str(ve)}]
    except Exception as e:
        logger.error(
            f"[QueryResult RequestID: {cmd.request_id}] Unexpected error in similarity search: {e}", exc_info=True
        )
        raise


@system(on_event=ImageAssetDetected)
async def process_image_metadata_system(
    event: ImageAssetDetected,
    transaction: WorldTransaction,
    world: World,
) -> None:
    """Listens for an image asset being detected and extracts its metadata."""
    logger.info(f"Processing image metadata for entity {event.entity.id}")

    try:
        # Skip Logic
        existing_component = await transaction.get_component(event.entity.id, ImageDimensionsComponent)
        if existing_component:
            logger.info(f"Entity {event.entity.id} already has ImageDimensionsComponent. Skipping.")
            return

        # Get file path from file_id
        file_path = await file_operations.get_file_path_by_id(world, transaction, event.file_id)
        if not file_path:
            logger.warning(
                f"Could not find file path for file_id {event.file_id} on entity {event.entity.id}. Cannot process image metadata."
            )
            return

        # Extract metadata
        with Image.open(file_path) as img:
            width, height = img.size

        # Add component
        dimensions_component = ImageDimensionsComponent()
        dimensions_component.width_pixels = width
        dimensions_component.height_pixels = height
        await transaction.add_component_to_entity(event.entity.id, dimensions_component)

        logger.info(f"Successfully added ImageDimensionsComponent to entity {event.entity.id}.")

    except Exception as e:
        logger.error(
            f"Failed during image metadata processing for entity {event.entity.id}: {e}",
            exc_info=True,
        )
        raise
