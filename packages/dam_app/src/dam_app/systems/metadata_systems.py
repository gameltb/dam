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
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional

from dam.core.config import WorldConfig
from dam.core.systems import system
from dam.core.transaction import EcsTransaction
from dam.models.metadata.exiftool_metadata_component import ExiftoolMetadataComponent
from dam_fs.models.file_location_component import FileLocationComponent
from dam_fs.utils.url_utils import get_local_path_for_url

from ..commands import ExtractExifMetadataCommand

logger = logging.getLogger(__name__)


class ExifTool:
    """A class to interact with a persistent exiftool process."""

    def __init__(self, executable: str = "exiftool"):
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
        ]

    async def start(self):
        """Starts the persistent exiftool process."""
        if self.process and self.process.returncode is None:
            return

        exiftool_path = shutil.which(self.executable)
        if not exiftool_path:
            raise FileNotFoundError(f"{self.executable} not found in PATH.")

        command = [exiftool_path, "-stay_open", "True", "-@", "-"]
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info("ExifTool process started.")

    async def stop(self):
        """Stops the persistent exiftool process and the thread pool."""
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

    def _sanitize_extension(self, extension: str) -> str:
        """Removes potentially unsafe characters from a file extension."""
        return "".join(char for char in extension if char.isalnum())

    def _write_stream_to_fifo(self, stream: BinaryIO, fifo_path: Path):
        """Writes the content of a stream to a FIFO."""
        try:
            with open(fifo_path, "wb") as fifo:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    fifo.write(chunk)
        except BrokenPipeError as e:
            # ExifTool may close fifo early
            pass
        except Exception as e:
            logger.error(f"Error writing to FIFO: {e}", exc_info=True)

    async def get_metadata(
        self,
        filepath: Optional[Path] = None,
        stream: Optional[BinaryIO] = None,
    ) -> Dict[str, Any] | None:
        """
        Extracts metadata from a file or stream using the persistent exiftool process.
        Returns None if an error occurs.
        """
        if not self.process or self.process.returncode is not None:
            await self.start()

        if not self.process or not self.process.stdin or not self.process.stdout:
            logger.error("ExifTool process is not running or pipes are not available.")
            return None

        loop = asyncio.get_running_loop()

        current_stream = None
        close_stream = False
        if stream:
            current_stream = stream
        elif filepath:
            try:
                current_stream = open(filepath, "rb")
                close_stream = True
            except FileNotFoundError:
                logger.error(f"File not found at {filepath}")
                return None

        if not current_stream:
            logger.error("Either filepath or stream must be provided to get_metadata.")
            return None

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                fifo_path = Path(temp_dir) / "exiftool_fifo"
                os.mkfifo(fifo_path)

                writer_future = loop.run_in_executor(
                    self.executor, self._write_stream_to_fifo, current_stream, fifo_path
                )

                command_parts = self.exiftool_args.copy()

                command_parts.append(str(fifo_path))

                command_bytes = b"\n".join(part.encode() for part in command_parts) + b"\n-execute\n"

                self.process.stdin.write(command_bytes)
                await self.process.stdin.drain()
                await writer_future

            output_bytes = await self.process.stdout.readuntil(b"{ready}\n")
            output_str = output_bytes.decode("utf-8", errors="ignore").strip()
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
                logger.warning(f"Exiftool output was not a list with one element: {json_str[:200]}")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from exiftool: {e}. Output: {json_str[:200]}")
                return None
        finally:
            if close_stream and current_stream:
                current_stream.close()


exiftool_instance = ExifTool()


@system(on_command=ExtractExifMetadataCommand)
async def extract_metadata_command_handler(
    cmd: ExtractExifMetadataCommand,
    transaction: EcsTransaction,
    world_config: WorldConfig,
):
    entity_id = cmd.entity_id
    logger.debug(f"Processing entity ID {entity_id} for metadata extraction.")

    exiftool_data = None

    if cmd.stream:
        logger.debug(f"Using provided stream for entity {entity_id}.")
        exiftool_data = await exiftool_instance.get_metadata(stream=cmd.stream)
    else:
        logger.debug(f"No stream provided for entity {entity_id}, finding file on disk.")
        all_locations = await transaction.get_components(entity_id, FileLocationComponent)
        if not all_locations:
            logger.warning(f"No FileLocationComponent found for Entity ID {entity_id}. Cannot extract metadata.")
            return

        filepath_to_process = None
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

        exiftool_data = await exiftool_instance.get_metadata(filepath=filepath_to_process)

    if exiftool_data:
        exif_comp = ExiftoolMetadataComponent(raw_exif_json=exiftool_data)
        await transaction.add_or_update_component(entity_id, exif_comp)
        logger.info(f"Added or updated ExiftoolMetadataComponent for Entity ID {entity_id}")
    else:
        logger.info(f"No metadata extracted by Exiftool for Entity ID {entity_id}")

    logger.info(f"Metadata extraction finished for entity {entity_id}.")
