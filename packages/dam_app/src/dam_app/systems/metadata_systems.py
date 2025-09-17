# pyright: ignore
"""
This module defines systems related to metadata extraction for assets.

Systems in this module are responsible for processing entities to extract and store detailed
metadata such as dimensions, duration, frame counts, audio properties, etc.,
using tools like the Hachoir library and exiftool.
"""

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from dam.core.config import WorldConfig
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.models.metadata.exiftool_metadata_component import ExiftoolMetadataComponent
from dam_fs.functions import file_operations
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.utils.url_utils import get_local_path_for_url

from ..commands import ExtractExifMetadataCommand

logger = logging.getLogger(__name__)


async def _run_exiftool_subprocess(filepath: Path) -> Dict[str, Any] | None:
    """
    Runs exiftool on the given filepath and returns the JSON output.
    Returns None if exiftool is not found or if an error occurs.
    """
    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        logger.warning("exiftool command not found in PATH. Skipping exiftool metadata extraction.")
        return None

    command = [exiftool_path, "-json", "-G", str(filepath)]
    try:
        process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(
                f"Exiftool error for {filepath} (return code {process.returncode}): {stderr.decode(errors='ignore')}"
            )
            return None

        try:
            # exiftool outputs a list with a single JSON object
            data = json.loads(stdout)
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            logger.warning(
                f"Exiftool output for {filepath} was not a list with one element: {stdout.decode(errors='ignore')[:200]}"
            )
            return None
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON from exiftool for {filepath}: {e}. Output: {stdout.decode(errors='ignore')[:200]}"
            )
            return None

    except Exception as e:
        logger.error(f"Exception running exiftool for {filepath}: {e}", exc_info=True)
        return None


async def _extract_metadata_with_exiftool_async(filepath_on_disk: Path) -> Dict[str, Any] | None:
    """
    Asynchronously calls exiftool metadata extraction.
    """
    return await _run_exiftool_subprocess(filepath_on_disk)


@system(on_command=ExtractExifMetadataCommand)
async def extract_metadata_command_handler(
    cmd: ExtractExifMetadataCommand,
    transaction: EcsTransaction,
    world_config: WorldConfig,
):
    entity_id = cmd.entity_id
    logger.debug(f"Processing entity ID {entity_id} for metadata extraction.")

    filepath_to_process: Optional[Path] = None
    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None

    try:
        if cmd.stream:
            logger.debug(f"Using provided stream for entity {entity_id}.")
            # exiftool needs a file on disk, so we write the stream to a temp file
            temp_dir = tempfile.TemporaryDirectory()
            temp_path = Path(temp_dir.name) / "temp_asset"
            with open(temp_path, "wb") as f:
                f.write(cmd.stream.read())
            filepath_to_process = temp_path
        else:
            logger.debug(f"No stream provided for entity {entity_id}, finding file on disk.")
            all_locations = await transaction.get_components(entity_id, FileLocationComponent)
            if not all_locations:
                logger.warning(f"No FileLocationComponent found for Entity ID {entity_id}. Cannot extract metadata.")
                return

            for loc in all_locations:
                try:
                    potential_path = get_local_path_for_url(loc.url)
                    if potential_path and await asyncio.to_thread(potential_path.is_file):
                        filepath_to_process = potential_path
                        break
                except (ValueError, FileNotFoundError) as e:
                    logger.debug(f"Could not resolve or find file for URL '{loc.url}' for entity {entity_id}: {e}")
                    continue

        if not filepath_to_process:
            logger.error(
                f"Filepath for Entity ID {entity_id} does not exist or could not be determined. Cannot extract metadata."
            )
            return

        mime_type = await file_operations.get_mime_type_async(filepath_to_process)
        logger.info(f"Extracting metadata from {filepath_to_process} for Entity ID {entity_id} (MIME: {mime_type})")

        logger.info(f"Attempting Exiftool metadata extraction for {filepath_to_process} (Entity ID {entity_id})")
        exiftool_data = await _extract_metadata_with_exiftool_async(filepath_to_process)

        if exiftool_data:
            exif_comp = ExiftoolMetadataComponent(raw_exif_json=exiftool_data)
            await transaction.add_or_update_component(entity_id, exif_comp)
            logger.info(f"Added or updated ExiftoolMetadataComponent for Entity ID {entity_id}")
        else:
            logger.info(f"No metadata extracted by Exiftool for {filepath_to_process} (Entity ID {entity_id})")

        logger.info(f"Metadata extraction finished for entity {entity_id}.")
    finally:
        if temp_dir:
            temp_dir.cleanup()
