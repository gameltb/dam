import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import py7zr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dam.models.core import Entity
from dam.models.hashes import (
    ContentHashCRC32Component,
    ContentHashMD5Component,
    ContentHashSHA1Component,
    ContentHashSHA256Component,
)
from dam.models.metadata import PSPSFOMetadataComponent, PspSfoRawMetadataComponent
from dam.services import ecs_service, hashing_service, psp_iso_service


async def _process_iso_file(session: AsyncSession, file_path: Path, file_stream: BytesIO) -> None:
    """Helper function to process a single ISO stream."""

    # Calculate hashes
    file_stream.seek(0)
    hash_algorithms = ["md5", "sha1", "sha256", "crc32"]
    hashes = hashing_service.calculate_hashes_from_stream(file_stream, hash_algorithms)
    if not hashes:
        return

    md5_hash = bytes.fromhex(hashes["md5"])

    # Check for duplicates
    stmt = select(Entity.id).join(ContentHashMD5Component).where(ContentHashMD5Component.hash_value == md5_hash)
    result = await session.execute(stmt)
    if result.scalars().first() is not None:
        return

    # Create entity and components
    entity = await ecs_service.create_entity(session)

    # Add hash components
    await ecs_service.add_component_to_entity(session, entity.id, ContentHashMD5Component(hash_value=md5_hash))
    await ecs_service.add_component_to_entity(
        session, entity.id, ContentHashSHA1Component(hash_value=bytes.fromhex(hashes["sha1"]))
    )
    await ecs_service.add_component_to_entity(
        session, entity.id, ContentHashSHA256Component(hash_value=bytes.fromhex(hashes["sha256"]))
    )
    await ecs_service.add_component_to_entity(
        session, entity.id, ContentHashCRC32Component(hash_value=hashes["crc32"].to_bytes(4, "big"))
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
            await ecs_service.add_component_to_entity(session, entity.id, sfo_component)

            sfo_raw_component = PspSfoRawMetadataComponent(metadata_json=sfo_metadata)
            await ecs_service.add_component_to_entity(session, entity.id, sfo_raw_component)


async def ingest_psp_isos_from_directory(
    session: AsyncSession,
    directory: str,
    passwords: Optional[List[str]] = None,
):
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
