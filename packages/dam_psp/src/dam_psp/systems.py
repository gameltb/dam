import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import py7zr
from dam.core.commands import AddHashesFromStreamCommand
from dam.core.config import WorldConfig
from dam.core.systems import handles_command, listens_for
from dam.core.transaction import EcsTransaction
from dam.core.world import World
from dam.models.core import Entity
from dam.models.hashes import (
    ContentHashMD5Component,
)
from dam.utils.hash_utils import HashAlgorithm, calculate_hashes_from_stream
from dam_fs.functions import file_operations
from sqlalchemy import select

from dam_psp.psp_iso_functions import process_iso_stream

from .commands import IngestPspIsosCommand
from .events import PspIsoAssetDetected
from .models import PSPSFOMetadataComponent, PspSfoRawMetadataComponent

logger = logging.getLogger(__name__)


@listens_for(PspIsoAssetDetected)
async def process_psp_iso_system(
    event: PspIsoAssetDetected,
    transaction: EcsTransaction,
    world_config: WorldConfig,
):
    """
    Listens for a PSP ISO asset being detected and extracts its SFO metadata.
    """
    logger.info(f"Processing PSP ISO metadata for entity {event.entity.id}")

    try:
        # Skip Logic
        existing_component = await transaction.get_component(event.entity.id, PSPSFOMetadataComponent)
        if existing_component:
            logger.info(f"Entity {event.entity.id} already has PSPSFOMetadataComponent. Skipping.")
            return

        # Get file path from file_id
        file_path = await file_operations.get_file_path_by_id(transaction, event.file_id, world_config)
        if not file_path:
            logger.warning(
                f"Could not find file path for file_id {event.file_id} on entity {event.entity.id}. Cannot process ISO."
            )
            return

        # Extract metadata
        with open(file_path, "rb") as f:
            sfo = process_iso_stream(f)

        if sfo and sfo.data:
            sfo_metadata = sfo.data
            sfo_component = PSPSFOMetadataComponent(
                app_ver=sfo_metadata.get("APP_VER"),
                bootable=sfo_metadata.get("BOOTABLE"),
                category=sfo_metadata.get("CATEGORY"),
                disc_id=sfo_metadata.get("DISC_ID"),
                disc_version=sfo_metadata.get("DISC_VERSION"),
                parental_level=sfo_metadata.get("PARENTAL_LEVEL"),
                psp_system_ver=sfo_metadata.get("PSP_SYSTEM_VER"),
                title=sfo_metadata.get("TITLE"),
            )
            await transaction.add_component_to_entity(event.entity.id, sfo_component)

            sfo_raw_component = PspSfoRawMetadataComponent(metadata_json=sfo_metadata)
            await transaction.add_component_to_entity(event.entity.id, sfo_raw_component)

            logger.info(f"Successfully added PSPSFOMetadataComponent to entity {event.entity.id}.")
        else:
            logger.warning(f"Could not extract SFO metadata from ISO for entity {event.entity.id}.")

    except Exception as e:
        logger.error(
            f"Failed during PSP ISO metadata processing for entity {event.entity.id}: {e}",
            exc_info=True,
        )
        raise


async def _process_iso_file(world: World, transaction: EcsTransaction, file_path: Path, file_stream: BytesIO) -> None:
    """Helper function to process a single ISO stream."""

    # Calculate only one hash for duplicate check
    file_stream.seek(0)
    hashes = calculate_hashes_from_stream(file_stream, {HashAlgorithm.MD5})
    if not hashes:
        return

    md5_hash = hashes[HashAlgorithm.MD5]

    # Check for duplicates
    stmt = select(Entity.id).join(ContentHashMD5Component).where(ContentHashMD5Component.hash_value == md5_hash)
    result = await transaction.session.execute(stmt)
    if result.scalars().first() is not None:
        return

    # Create entity
    entity = await transaction.create_entity()

    # Dispatch command to add all hashes
    file_stream.seek(0)
    add_hashes_command = AddHashesFromStreamCommand(
        entity_id=entity.id,
        stream=file_stream,
        algorithms={HashAlgorithm.MD5, HashAlgorithm.SHA1, HashAlgorithm.SHA256, HashAlgorithm.CRC32},
    )
    await world.dispatch_command(add_hashes_command)


    # Process SFO metadata
    file_stream.seek(0)
    try:
        sfo = process_iso_stream(file_stream)
    except Exception:
        sfo = None

    if sfo:
        # Add SFO metadata component
        sfo_metadata = sfo.data
        if sfo_metadata:
            sfo_component = PSPSFOMetadataComponent(
                app_ver=sfo_metadata.get("APP_VER"),
                bootable=sfo_metadata.get("BOOTABLE"),
                category=sfo_metadata.get("CATEGORY"),
                disc_id=sfo_metadata.get("DISC_ID"),
                disc_version=sfo_metadata.get("DISC_VERSION"),
                parental_level=sfo_metadata.get("PARENTAL_LEVEL"),
                psp_system_ver=sfo_metadata.get("PSP_SYSTEM_VER"),
                title=sfo_metadata.get("TITLE"),
            )
            await transaction.add_component_to_entity(entity.id, sfo_component)

            sfo_raw_component = PspSfoRawMetadataComponent(metadata_json=sfo_metadata)
            await transaction.add_component_to_entity(entity.id, sfo_raw_component)


@handles_command(IngestPspIsosCommand)
async def ingest_psp_isos_from_directory_system(
    cmd: IngestPspIsosCommand,
    world: World,
    transaction: EcsTransaction,
):
    """
    Scans a directory for PSP ISOs and archives, processes them, and stores them in the database.
    """
    if cmd.passwords is None:
        passwords = [None]  # Try with no password first
    else:
        passwords = [None] + cmd.passwords

    for root, _, files in os.walk(cmd.directory):
        for filename in files:
            file_path = Path(root) / filename
            ext = file_path.suffix.lower()

            if ext == ".iso":
                with open(file_path, "rb") as f:
                    await _process_iso_file(world, transaction, file_path, BytesIO(f.read()))

            elif ext == ".zip":
                for password in passwords:
                    try:
                        with zipfile.ZipFile(file_path, "r") as zf:
                            if password:
                                zf.setpassword(password.encode())
                            for member_name in zf.namelist():
                                if member_name.lower().endswith(".iso"):
                                    with zf.open(member_name) as iso_file:
                                        await _process_iso_file(
                                            world,
                                            transaction,
                                            file_path / member_name,
                                            BytesIO(iso_file.read()),
                                        )
                        break
                    except (RuntimeError, zipfile.BadZipFile):
                        continue

            elif ext == ".7z":
                for password in passwords:
                    try:
                        with py7zr.SevenZipFile(file_path, mode="r", password=password) as szf:
                            for member_name, bio in szf.read().items():
                                if member_name.lower().endswith(".iso"):
                                    await _process_iso_file(world, transaction, file_path / member_name, bio)
                        break
                    except py7zr.exceptions.PasswordRequired:
                        continue
                    except py7zr.exceptions.Bad7zFile:
                        continue
                    except Exception:
                        continue
