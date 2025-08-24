import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import py7zr
from dam.core.config import WorldConfig
from dam.core.systems import listens_for
from dam.core.transaction import EcsTransaction
from dam.models.core import Entity
from dam.models.hashes import (
    ContentHashCRC32Component,
    ContentHashMD5Component,
    ContentHashSHA1Component,
    ContentHashSHA256Component,
)
from dam.services import ecs_service, hashing_service
from dam_fs.services import file_operations
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import service as psp_iso_service
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
            sfo = psp_iso_service.process_iso_stream(f)

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


async def _process_iso_file(transaction: EcsTransaction, file_path: Path, file_stream: BytesIO) -> None:
    """Helper function to process a single ISO stream."""

    # Calculate hashes
    file_stream.seek(0)
    hash_algorithms = ["md5", "sha1", "sha256", "crc32"]
    hashes = hashing_service.calculate_hashes_from_stream(file_stream, hash_algorithms)
    if not hashes:
        return

    md5_hash = bytes.fromhex(hashes["md5"])

    # Check for duplicates
    # This is a read operation, so it's fine to use the session directly.
    stmt = select(Entity.id).join(ContentHashMD5Component).where(ContentHashMD5Component.hash_value == md5_hash)
    result = await transaction.session.execute(stmt)
    if result.scalars().first() is not None:
        return

    # Create entity and components
    entity = await transaction.create_entity()

    # Add hash components
    await transaction.add_component_to_entity(entity.id, ContentHashMD5Component(hash_value=md5_hash))
    await transaction.add_component_to_entity(
        entity.id, ContentHashSHA1Component(hash_value=bytes.fromhex(hashes["sha1"]))
    )
    await transaction.add_component_to_entity(
        entity.id, ContentHashSHA256Component(hash_value=bytes.fromhex(hashes["sha256"]))
    )
    await transaction.add_component_to_entity(
        entity.id, ContentHashCRC32Component(hash_value=hashes["crc32"].to_bytes(4, "big"))
    )

    # Process SFO metadata
    file_stream.seek(0)
    try:
        sfo = psp_iso_service.process_iso_stream(file_stream)
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


async def ingest_psp_isos_from_directory(
    session: AsyncSession,
    directory: str,
    passwords: Optional[List[str]] = None,
):
    # TODO: This function creates its own session and is not part of the
    # main transactional event/command bus. It should be refactored to
    # be a command handler that uses the EcsTransaction object.
    """
    Scans a directory for PSP ISOs and archives, processes them, and stores them in the database.
    """
    if passwords is None:
        passwords = [None]  # Try with no password first
    else:
        passwords.insert(0, None)  # Also try with no password first

    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = Path(root) / filename
            ext = file_path.suffix.lower()

            if ext == ".iso":
                with open(file_path, "rb") as f:
                    await _process_iso_file(session, file_path, BytesIO(f.read()))

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
                                            session,
                                            file_path / member_name,
                                            BytesIO(iso_file.read()),
                                        )
                        break  # Correct password found
                    except (RuntimeError, zipfile.BadZipFile):
                        continue  # Wrong password, try next

            elif ext == ".7z":
                for password in passwords:
                    try:
                        with py7zr.SevenZipFile(file_path, mode="r", password=password) as szf:
                            for member_name, bio in szf.read().items():
                                if member_name.lower().endswith(".iso"):
                                    await _process_iso_file(session, file_path / member_name, bio)
                        break  # Correct password found
                    except py7zr.exceptions.PasswordRequired:
                        continue
                    except py7zr.exceptions.Bad7zFile:
                        continue
                    except Exception:
                        continue
