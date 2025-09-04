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
from pathlib import Path
from typing import Any, Dict

from dam.core.config import WorldConfig
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.models.metadata.exiftool_metadata_component import ExiftoolMetadataComponent
from dam_fs.functions import file_operations
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.utils.url_utils import get_local_path_for_url

from ..commands import ExtractMetadataCommand

try:
    from hachoir.core import config as HachoirConfig
    from hachoir.metadata import extractMetadata
    from hachoir.parser import createParser

    HachoirConfig.quiet = True
except ImportError:
    createParser = None
    extractMetadata = None

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


def _has_hachoir_metadata(md, key: str) -> bool:
    """Safely checks if Hachoir metadata object has a given key."""
    try:
        return md.has(key)
    except (KeyError, ValueError):
        return False


def _get_hachoir_metadata(md, key: str, default=None) -> Any:
    """Safely gets a value from Hachoir metadata object for a given key."""
    try:
        if md.has(key):
            return md.get(key)
    except (KeyError, ValueError):
        pass
    return default


# This is the internal synchronous parsing function
def _parse_metadata_sync_worker(filepath_str: str) -> Any | None:
    if not createParser or not extractMetadata:
        logger.info("Hachoir library not available. Cannot perform sync extraction.")
        return None

    try:
        file_size = Path(filepath_str).stat().st_size
        logger.info(f"Hachoir worker attempting to parse: {filepath_str}, size: {file_size} bytes")
    except Exception as e_stat:
        logger.error(f"Hachoir worker: Error stating file {filepath_str}: {e_stat}")
        # Proceed to createParser, it will likely fail and log its own error.

    parser = None  # Initialize parser to None
    try:
        parser = createParser(filepath_str)
        if not parser:
            logger.warning(f"Hachoir could not create a parser for file: {filepath_str} (createParser returned None)")
            return None

        with parser:  # parser is guaranteed to be non-None here
            try:
                metadata = extractMetadata(parser)
                return metadata
            except Exception as e_extract:  # More specific exception variable
                logger.error(f"Hachoir failed to extract metadata for {filepath_str}: {e_extract}", exc_info=True)
                return None
    except HachoirConfig.HachoirError as e_hachoir:  # Catch Hachoir specific errors like NullStreamError
        # HachoirError is a base class for many hachoir exceptions including NullStreamError
        logger.warning(f"Hachoir error during parsing or metadata extraction for {filepath_str}: {e_hachoir}")
        return None
    except Exception as e_general:  # Catch any other unexpected errors during createParser or context management
        logger.error(f"Unexpected error with Hachoir for {filepath_str}: {e_general}", exc_info=True)
        return None


async def _extract_metadata_with_hachoir_sync(filepath_on_disk: Path) -> Any | None:
    """
    Asynchronously calls the synchronous Hachoir metadata extraction.
    """
    return await asyncio.to_thread(_parse_metadata_sync_worker, str(filepath_on_disk))


@system(on_command=ExtractMetadataCommand)
async def extract_metadata_command_handler(
    cmd: ExtractMetadataCommand,
    transaction: EcsTransaction,
    world_config: WorldConfig,
):
    if not createParser or not extractMetadata:  # Hachoir check
        logger.warning("Hachoir library not installed. Skipping metadata extraction.")
        return

    entity_id = cmd.entity_id
    logger.debug(f"Processing entity ID {entity_id} for metadata extraction.")

    all_locations = await transaction.get_components(entity_id, FileLocationComponent)
    if not all_locations:
        logger.warning(f"No FileLocationComponent found for Entity ID {entity_id}. Cannot extract metadata.")
        return

    filepath_on_disk: Path | None = None
    for loc in all_locations:
        try:
            potential_path = get_local_path_for_url(loc.url)
            if potential_path and await asyncio.to_thread(potential_path.is_file):
                filepath_on_disk = potential_path
                break
        except (ValueError, FileNotFoundError) as e:
            logger.debug(f"Could not resolve or find file for URL '{loc.url}' for entity {entity_id}: {e}")
            continue

    if not filepath_on_disk:
        logger.error(
            f"Filepath for Entity ID {entity_id} does not exist or could not be determined. Cannot extract metadata."
        )
        return

    mime_type = await file_operations.get_mime_type_async(filepath_on_disk)
    logger.info(f"Extracting metadata from {filepath_on_disk} for Entity ID {entity_id} (MIME: {mime_type})")
    hachoir_metadata = await _extract_metadata_with_hachoir_sync(filepath_on_disk)

    if hachoir_metadata:
        keys_to_log = []
        if hasattr(hachoir_metadata, "keys") and callable(hachoir_metadata.keys):
            try:
                keys_to_log = list(hachoir_metadata.keys())
            except Exception:
                keys_to_log = ["<error reading Hachoir metadata keys>"]
        logger.debug(
            f"Hachoir metadata (type: {type(hachoir_metadata).__name__}) keys for {filepath_on_disk}: {keys_to_log}"
        )
        # Hachoir processing logic would go here
    else:
        logger.info(f"No metadata extracted by Hachoir for {filepath_on_disk} (Entity ID {entity_id})")

    logger.info(f"Attempting Exiftool metadata extraction for {filepath_on_disk} (Entity ID {entity_id})")
    exiftool_data = await _extract_metadata_with_exiftool_async(filepath_on_disk)

    if exiftool_data:
        if not await transaction.get_component(entity_id, ExiftoolMetadataComponent):
            exif_comp = ExiftoolMetadataComponent(raw_exif_json=exiftool_data)
            await transaction.add_component_to_entity(entity_id, exif_comp)
            logger.info(f"Added ExiftoolMetadataComponent for Entity ID {entity_id}")
        else:
            logger.info(f"ExiftoolMetadataComponent already exists for Entity ID {entity_id}, not adding duplicate.")
    else:
        logger.info(f"No metadata extracted by Exiftool for {filepath_on_disk} (Entity ID {entity_id})")

    logger.info(f"Metadata extraction finished for entity {entity_id}.")
