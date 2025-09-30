import hashlib
import os
import struct
import zlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from dam.core.types import CallableStreamProvider
from dam.models.core.entity import Entity
from dam.system_events.entity_events import NewEntityCreatedEvent
from pytest_mock import MockerFixture

from dam_psp.commands import IngestCsoCommand
from dam_psp.cso_support import CsoDecompressor
from dam_psp.models import CsoParentIsoComponent, IngestedCsoComponent
from dam_psp.systems import ingest_cso_handler

# CSO constants
CISO_MAGIC = 0x4F534943  # CISO
CISO_HEADER_SIZE = 0x18  # 24
CISO_BLOCK_SIZE = 0x800  # 2048
CISO_HEADER_FMT = "<LLQLBBxx"  # Little endian


def create_dummy_iso(filepath: Path, size_in_mb: int):
    """Creates a dummy ISO file with patterned data."""
    with open(filepath, "wb") as f:
        for i in range(size_in_mb * 1024 // (CISO_BLOCK_SIZE // 1024)):
            block_num_bytes = struct.pack("<I", i)
            padding = CISO_BLOCK_SIZE - len(block_num_bytes)
            f.write(block_num_bytes + b"\xaa" * padding)


def compress_iso(infile: Path, outfile: Path):
    """Compresses an ISO file to the CSO format."""
    with open(outfile, "wb") as fout:
        with open(infile, "rb") as fin:
            fin.seek(0, os.SEEK_END)
            file_size = fin.tell()
            fin.seek(0, os.SEEK_SET)

            ciso = {
                "magic": CISO_MAGIC,
                "ver": 1,
                "block_size": CISO_BLOCK_SIZE,
                "total_bytes": file_size,
                "total_blocks": file_size // CISO_BLOCK_SIZE,
                "align": 0,
            }

            fout.write(
                struct.pack(
                    CISO_HEADER_FMT,
                    ciso["magic"],
                    CISO_HEADER_SIZE,
                    ciso["total_bytes"],
                    ciso["block_size"],
                    ciso["ver"],
                    ciso["align"],
                )
            )

            block_index = [0] * (ciso["total_blocks"] + 1)
            for _ in range(ciso["total_blocks"] + 1):
                fout.write(struct.pack("<I", 0))

            write_pos = fout.tell()

            for block in range(ciso["total_blocks"]):
                block_index[block] = write_pos >> ciso["align"]
                raw_data = fin.read(ciso["block_size"])

                compress_obj = zlib.compressobj(level=9, wbits=-15)
                compressed_data = compress_obj.compress(raw_data) + compress_obj.flush()

                if len(compressed_data) >= len(raw_data):
                    block_index[block] |= 0x80000000
                    writable_data = raw_data
                else:
                    writable_data = compressed_data

                fout.write(writable_data)
                write_pos += len(writable_data)

            block_index[ciso["total_blocks"]] = write_pos >> ciso["align"]

            fout.seek(CISO_HEADER_SIZE)
            for index_val in block_index:
                fout.write(struct.pack("<I", index_val))


@pytest.fixture
def cso_test_files(tmp_path: Path) -> tuple[Path, Path]:
    """Generates a dummy ISO and CSO file for testing and returns their paths."""
    iso_path = tmp_path / "test.iso"
    cso_path = tmp_path / "test.cso"
    create_dummy_iso(iso_path, 1)
    compress_iso(iso_path, cso_path)
    return iso_path, cso_path


def test_cso_decompressor(cso_test_files: tuple[Path, Path]):
    """
    Tests that the CsoDecompressor correctly decompresses a CSO file.
    """
    iso_path, cso_path = cso_test_files
    iso_content = iso_path.read_bytes()
    with open(cso_path, "rb") as f:
        decompressor = CsoDecompressor(f)
        decompressed_data = decompressor.read()

    assert len(decompressed_data) == len(iso_content)
    assert hashlib.sha256(decompressed_data).hexdigest() == hashlib.sha256(iso_content).hexdigest()


def test_cso_decompressor_seek_and_read(cso_test_files: tuple[Path, Path]):
    """
    Tests that seeking and reading from the decompressor works correctly.
    """
    iso_path, cso_path = cso_test_files
    iso_content = iso_path.read_bytes()
    with open(cso_path, "rb") as f:
        decompressor = CsoDecompressor(f)

        # Seek to a position and read a chunk
        decompressor.seek(2048)
        data_chunk = decompressor.read(1024)

        assert data_chunk == iso_content[2048:3072]

        # Seek to the end
        decompressor.seek(0, os.SEEK_END)
        assert decompressor.tell() == len(iso_content)

        # Read from the current position (end of file)
        assert decompressor.read() == b""


@pytest.mark.asyncio
async def test_ingest_cso_handler_system(mocker: MockerFixture, cso_test_files: tuple[Path, Path]):
    """
    Tests the ingest_cso_handler system using mocks.
    """
    # 1. Setup
    cso_entity_id = 1
    virtual_iso_entity_id = 2
    iso_path, cso_path = cso_test_files

    mock_transaction = AsyncMock()
    mock_transaction.add_or_update_component = AsyncMock()

    mock_world = MagicMock()

    # Mock the return of GetOrCreateEntityFromStreamCommand
    mock_iso_entity = Entity()
    mock_iso_entity.id = virtual_iso_entity_id
    get_or_create_result = MagicMock()
    get_or_create_result.get_one_value = AsyncMock(return_value=(mock_iso_entity, True))

    # Mock the return of GetAssetFilenamesCommand
    get_filenames_result = MagicMock()
    get_filenames_result.get_all_results_flat = AsyncMock(return_value=[cso_path.name])

    def dispatch_side_effect(cmd: object) -> MagicMock:
        if "GetOrCreateEntityFromStreamCommand" in str(type(cmd)):
            return get_or_create_result
        if "GetAssetFilenamesCommand" in str(type(cmd)):
            return get_filenames_result
        return MagicMock()

    mock_world.dispatch_command.side_effect = dispatch_side_effect

    # Create the command to be tested
    with open(cso_path, "rb") as cso_stream:
        cmd = IngestCsoCommand(entity_id=cso_entity_id, stream_provider=CallableStreamProvider(lambda: cso_stream))

        # 2. Execute
        events = [event async for event in ingest_cso_handler(cmd, mock_transaction, mock_world)]

    # 3. Assert
    # Check that components were added correctly
    add_comp_calls = mock_transaction.add_or_update_component.call_args_list
    assert len(add_comp_calls) == 2

    # Check CsoParentIsoComponent on virtual ISO
    parent_iso_comp_call = next(c for c in add_comp_calls if isinstance(c.args[1], CsoParentIsoComponent))
    assert parent_iso_comp_call.args[0] == virtual_iso_entity_id
    assert parent_iso_comp_call.args[1].cso_entity_id == cso_entity_id

    # Check IngestedCsoComponent on CSO file
    ingested_cso_comp_call = next(c for c in add_comp_calls if isinstance(c.args[1], IngestedCsoComponent))
    assert ingested_cso_comp_call.args[0] == cso_entity_id

    # Check that a NewEntityCreatedEvent was yielded
    new_entity_events = [e for e in events if isinstance(e, NewEntityCreatedEvent)]
    assert len(new_entity_events) == 1
    assert new_entity_events[0].entity_id == virtual_iso_entity_id
    assert new_entity_events[0].filename == cso_path.with_suffix(".iso").name
