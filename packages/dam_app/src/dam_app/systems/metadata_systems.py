# pyright: ignore
"""
Defines systems related to metadata extraction for assets.

Systems in this module are responsible for processing entities to extract and store detailed
metadata such as dimensions, duration, frame counts, audio properties, etc.,
using tools like the Hachoir library and exiftool.
"""

import asyncio
import io
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from ctypes import CDLL, c_int
from pathlib import Path
from typing import Any, BinaryIO

import aiofiles
from dam.core.systems import system
from dam.core.transaction import WorldTransaction
from dam.core.world import World
from dam.models.metadata.exiftool_metadata_component import ExiftoolMetadataComponent
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.utils.url_utils import get_local_path_for_url

from ..commands import (
    CheckExifMetadataCommand,
    ExtractExifMetadataCommand,
    RemoveExifMetadataCommand,
)

logger = logging.getLogger(__name__)


@system(on_command=CheckExifMetadataCommand)
async def check_exif_metadata_handler(
    cmd: CheckExifMetadataCommand,
    transaction: WorldTransaction,
) -> bool:
    """Check if the ExiftoolMetadataComponent exists for the entity."""
    component = await transaction.get_component(cmd.entity_id, ExiftoolMetadataComponent)
    return component is not None


@system(on_command=RemoveExifMetadataCommand)
async def remove_exif_metadata_handler(
    cmd: RemoveExifMetadataCommand,
    transaction: WorldTransaction,
):
    """Remove the ExiftoolMetadataComponent from the entity."""
    component = await transaction.get_component(cmd.entity_id, ExiftoolMetadataComponent)
    if component:
        await transaction.remove_component(component)
        logger.info("Removed ExiftoolMetadataComponent from entity %d", cmd.entity_id)


class ExifTool:
    """A class to interact with a persistent exiftool process."""

    def __init__(self, executable: str = "exiftool"):
        """Initialize the ExifTool wrapper."""
        self.executable = executable
        self.process: asyncio.subprocess.Process | None = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.exiftool_args = [
            "-json",
            "-G0",
            "-a",
            "-s",
            "-e",
            "-b",
            "-x",
            "FILE:File*",
            "-x",
            "FILE:Directory",
            "-x",
            "File:MIMEType",
            "-x",
            "ExifTool:all",
            "-x",
            "Torrent:Pieces",
        ]

    async def start(self):
        """Start the persistent exiftool process."""
        if self.process and self.process.returncode is None:
            return

        exiftool_path = shutil.which(self.executable)
        if not exiftool_path:
            raise FileNotFoundError(f"{self.executable} not found in PATH.")

        command = [exiftool_path, "-stay_open", "True", "-@", "-"]

        preexec_fn = None
        if hasattr(os, "setsid"):

            def set_pdeathsig():
                try:
                    libc = CDLL("libc.so.6")
                    pr_set_pdeathsig = 1
                    libc.prctl(c_int(pr_set_pdeathsig), c_int(signal.SIGHUP))
                except (AttributeError, OSError) as e:
                    logger.warning("Failed to set PDEATHSIG: %s", e)

            preexec_fn = set_pdeathsig

        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=preexec_fn,
        )
        logger.info("ExifTool process started.")

    async def stop(self):
        """Stop the persistent exiftool process and the thread pool."""
        if self.process and self.process.returncode is None:
            if self.process.stdin:
                try:
                    self.process.stdin.write(b"-stay_open\nFalse\n")
                    await self.process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    logger.warning("Connection to exiftool process already closed.")
            await self.process.wait()
            self.process = None
            logger.info("ExifTool process stopped.")
        self.executor.shutdown(wait=False)

    def _write_stream_to_fifo(self, stream: BinaryIO, fifo_path: Path):
        """Write the content of a stream to a FIFO."""
        try:
            with fifo_path.open("wb") as fifo:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    fifo.write(chunk)
        except BrokenPipeError:
            pass
        except Exception:
            logger.exception("Error writing to FIFO")

    async def _execute_and_read_exiftool(self, fifo_path: Path) -> str:
        """Execute the exiftool command and read the output."""
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("ExifTool process is not running or pipes are not available.")

        command_parts = self.exiftool_args.copy()
        command_parts.append(str(fifo_path))
        command_bytes = b"\n".join(part.encode() for part in command_parts) + b"\n-execute\n"

        self.process.stdin.write(command_bytes)
        await self.process.stdin.drain()

        buffer = bytearray()
        separator = b"{ready}\n"
        while not buffer.endswith(separator):
            chunk = await self.process.stdout.read(4096)
            if not chunk:
                break
            buffer.extend(chunk)

        output_bytes = bytes(buffer)
        return output_bytes.decode("utf-8", errors="ignore").strip()

    def _parse_json_output(self, output_str: str) -> dict[str, Any] | None:
        """Parse the JSON output from exiftool."""
        json_str = output_str.rsplit("{ready}", 1)[0].strip()
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
            if isinstance(data, list) and len(data) > 0:
                metadata = data[0]
                if "SourceFile" in metadata:
                    del metadata["SourceFile"]
                return metadata
            logger.warning("Exiftool output was not a list with one element: %s", json_str[:200])
            return None
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from exiftool: %s. Output: %s", e, json_str[:200])
            return None

    async def get_metadata(
        self,
        filepath: Path | None = None,
        stream: BinaryIO | None = None,
    ) -> dict[str, Any] | None:
        """
        Extract metadata from a file or stream using the persistent exiftool process.

        Returns None if an error occurs.
        """
        if not self.process or self.process.returncode is not None:
            await self.start()

        current_stream = None
        close_stream = False
        if stream:
            current_stream = stream
        elif filepath:
            try:
                async with aiofiles.open(filepath, "rb") as f:
                    content = await f.read()
                current_stream = io.BytesIO(content)
                close_stream = True
            except FileNotFoundError:
                logger.error("File not found at %s", filepath)
                return None

        if not current_stream:
            logger.error("Either filepath or stream must be provided to get_metadata.")
            return None

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                fifo_path = Path(temp_dir) / "exiftool_fifo"
                os.mkfifo(fifo_path)

                loop = asyncio.get_running_loop()
                writer_future = loop.run_in_executor(
                    self.executor, self._write_stream_to_fifo, current_stream, fifo_path
                )
                output_str = await self._execute_and_read_exiftool(fifo_path)
                await writer_future

                return self._parse_json_output(output_str)
        finally:
            if close_stream and current_stream:
                current_stream.close()


exiftool_instance = ExifTool()


async def _get_filepath_for_extraction(transaction: WorldTransaction, entity_id: int) -> Path | None:
    """Find a readable local file path for a given entity."""
    all_locations = await transaction.get_components(entity_id, FileLocationComponent)
    if not all_locations:
        logger.warning("No FileLocationComponent found for Entity ID %d. Cannot extract metadata.", entity_id)
        return None

    for loc in all_locations:
        try:
            potential_path = get_local_path_for_url(loc.url)
            if potential_path and await asyncio.to_thread(potential_path.is_file):
                return potential_path
        except (ValueError, FileNotFoundError) as e:
            logger.debug("Could not resolve or find file for URL '%s' for entity %d: %s", loc.url, entity_id, e)
            continue

    logger.error(
        "Filepath for Entity ID %d does not exist or could not be determined. Cannot extract metadata.", entity_id
    )
    return None


@system(on_command=ExtractExifMetadataCommand)
async def extract_metadata_command_handler(
    cmd: ExtractExifMetadataCommand,
    transaction: WorldTransaction,
    world: World,
):
    """Handle the command to extract EXIF metadata from an entity."""
    entity_id = cmd.entity_id
    logger.debug("Processing entity ID %d for metadata extraction.", entity_id)

    exiftool_data = None
    provider = await cmd.get_stream_provider(world)
    if provider:
        async with provider.get_stream() as stream:
            logger.debug("Using provided stream for entity %d.", entity_id)
            exiftool_data = await exiftool_instance.get_metadata(stream=stream)
    else:
        logger.debug("No stream provider for entity %d, finding file on disk.", entity_id)
        filepath_to_process = await _get_filepath_for_extraction(transaction, entity_id)
        if filepath_to_process:
            exiftool_data = await exiftool_instance.get_metadata(filepath=filepath_to_process)

    if exiftool_data:
        exif_comp = ExiftoolMetadataComponent(raw_exif_json=exiftool_data)
        await transaction.add_or_update_component(entity_id, exif_comp)
        logger.info("Added or updated ExiftoolMetadataComponent for Entity ID %d", entity_id)
    else:
        logger.info("No metadata extracted by Exiftool for Entity ID %d", entity_id)

    logger.info("Metadata extraction finished for entity %d.", entity_id)
