from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Generator, Optional
from unittest.mock import AsyncMock

import pycdlib
import pytest
from dam.commands.asset_commands import (
    GetAssetFilenamesCommand,
    GetAssetStreamCommand,
)
from dam.core.world import World
from dam.events import AssetReadyForMetadataExtractionEvent
from dam_archive.models import ArchiveMemberComponent
from dam_fs.models import FilenameComponent
from pytest_mock import MockerFixture

from dam_psp import psp_iso_functions
from dam_psp.commands import ExtractPSPMetadataCommand
from dam_psp.models import PSPSFOMetadataComponent
from dam_psp.systems import (
    psp_iso_metadata_extraction_command_handler_system,
    psp_iso_metadata_extraction_event_handler_system,
)

# A more realistic, valid PARAM.SFO file content for testing
DUMMY_SFO_CONTENT = b"".join(
    [
        b"\x00PSF",  # Magic
        b"\x01\x01\x00\x00",  # Version 1.1
        b"\x34\x00\x00\x00",  # key_table_start (52)
        b"\x4c\x00\x00\x00",  # data_table_start (76)
        b"\x02\x00\x00\x00",  # table_entries (2)
        # Index Table (2 entries * 16 bytes = 32 bytes)
        # Entry 1: TITLE
        b"\x00\x00",  # key_offset
        b"\x04\x02",  # data_fmt (UTF8)
        b"\x0a\x00\x00\x00",  # data_len (10)
        b"\x0a\x00\x00\x00",  # data_max_len (10)
        b"\x00\x00\x00\x00",  # data_offset
        # Entry 2: DISC_ID
        b"\x06\x00",  # key_offset
        b"\x04\x02",  # data_fmt (UTF8)
        b"\x0b\x00\x00\x00",  # data_len (11)
        b"\x0b\x00\x00\x00",  # data_max_len (11)
        b"\x0a\x00\x00\x00",  # data_offset
        # Padding to key_table_start (52)
        b"\x00" * (52 - (20 + 32)),
        # Key Table
        b"TITLE\x00",
        b"DISC_ID\x00",
        # Padding to data_table_start (76)
        b"\x00" * (76 - (52 + 6 + 8)),
        # Data Table
        b"Test Game\x00",
        b"ULUS-12345\x00",
    ]
)


def create_dummy_iso_with_sfo() -> BytesIO:
    """Creates a dummy ISO 9660 image with a PARAM.SFO file in memory."""
    iso = pycdlib.PyCdlib()  # type: ignore[attr-defined]
    iso.new(interchange_level=1, joliet=True)  # type: ignore
    iso.add_directory("/PSP_GAME")  # type: ignore
    iso.add_fp(BytesIO(DUMMY_SFO_CONTENT), len(DUMMY_SFO_CONTENT), "/PSP_GAME/PARAM.SFO;1")  # type: ignore

    iso_fp = BytesIO()
    iso.write_fp(iso_fp)  # type: ignore
    iso.close()

    iso_fp.seek(0)
    return iso_fp


@pytest.fixture
def dummy_iso_stream() -> Generator[BytesIO, None, None]:
    """Pytest fixture to provide a dummy ISO stream."""
    yield create_dummy_iso_with_sfo()


def test_process_iso_stream_extracts_sfo_metadata(dummy_iso_stream: BytesIO) -> None:
    """
    Tests that process_iso_stream correctly extracts metadata from a dummy ISO.
    """
    sfo = psp_iso_functions.process_iso_stream(dummy_iso_stream)

    assert sfo is not None
    sfo_metadata = sfo.data
    assert sfo_metadata is not None
    assert sfo_metadata.get("TITLE") == "Test Game"
    assert sfo_metadata.get("DISC_ID") == "ULUS-12345"


def test_process_iso_stream_handles_non_iso_file() -> None:
    """
    Tests that process_iso_stream raises an IOError for non-ISO files.
    """
    non_iso_stream = BytesIO(b"this is not an iso file")
    with pytest.raises(IOError):
        psp_iso_functions.process_iso_stream(non_iso_stream)


@pytest.mark.asyncio
async def test_psp_iso_metadata_extraction_system(mocker: MockerFixture) -> None:
    """
    Tests the psp_iso_metadata_extraction_system with a mix of assets.
    """
    # 1. Setup
    # Create mock entities
    standalone_iso_entity_id = 1
    archived_iso_entity_id = 2
    non_iso_entity_id = 3

    # Mock transaction
    mock_transaction = AsyncMock()

    async def get_component_side_effect(entity_id: int, component_type: Any) -> Optional[Any]:
        if entity_id == standalone_iso_entity_id:
            if component_type == PSPSFOMetadataComponent:
                return None
            if component_type == ArchiveMemberComponent:
                return None
            if component_type == FilenameComponent:
                return FilenameComponent(filename="test.iso", first_seen_at=datetime.now(timezone.utc))
        elif entity_id == archived_iso_entity_id:
            if component_type == PSPSFOMetadataComponent:
                return None
            if component_type == ArchiveMemberComponent:
                return ArchiveMemberComponent(archive_entity_id=99, path_in_archive="game.iso", modified_at=None)
        elif entity_id == non_iso_entity_id:
            if component_type == PSPSFOMetadataComponent:
                return None
            if component_type == ArchiveMemberComponent:
                return None
            if component_type == FilenameComponent:
                return FilenameComponent(filename="text.txt", first_seen_at=datetime.now(timezone.utc))
        return None

    mock_transaction.get_component.side_effect = get_component_side_effect
    mock_transaction.add_or_update_component = AsyncMock()

    # Mock world
    mock_world = AsyncMock(spec=World)

    def dispatch_command_side_effect(command: Any) -> AsyncMock:
        mock_stream = AsyncMock()
        if isinstance(command, GetAssetFilenamesCommand):
            if command.entity_id == standalone_iso_entity_id:
                mock_stream.get_all_results_flat.return_value = ["test.iso"]
            elif command.entity_id == archived_iso_entity_id:
                mock_stream.get_all_results_flat.return_value = ["game.iso"]
            elif command.entity_id == non_iso_entity_id:
                mock_stream.get_all_results_flat.return_value = ["text.txt"]
        elif isinstance(command, ExtractPSPMetadataCommand):
            mock_stream.get_all_results.return_value = []
        else:
            mock_stream.get_all_results.return_value = []

        return mock_stream

    mock_world.dispatch_command.side_effect = dispatch_command_side_effect

    # Create event
    event = AssetReadyForMetadataExtractionEvent(
        entity_ids=[standalone_iso_entity_id, archived_iso_entity_id, non_iso_entity_id]
    )

    # 2. Execute event handler
    await psp_iso_metadata_extraction_event_handler_system(event, mock_transaction, mock_world)

    # 3. Assert event handler dispatched commands correctly
    # It should have been called 3 times for GetAssetFilenamesCommand and 2 times for ExtractPSPMetadataCommand
    assert mock_world.dispatch_command.call_count == 5
    dispatch_calls = mock_world.dispatch_command.call_args_list

    # Check that ExtractPSPMetadataCommand was dispatched for the two ISO entities
    extract_commands = [call.args[0] for call in dispatch_calls if isinstance(call.args[0], ExtractPSPMetadataCommand)]
    assert len(extract_commands) == 2
    assert extract_commands[0].entity_id == standalone_iso_entity_id
    assert extract_commands[1].entity_id == archived_iso_entity_id

    # 4. Execute command handler
    # Reset mock for world dispatch before command handler execution
    def dispatch_command_side_effect_for_stream(command: Any) -> AsyncMock:
        mock_stream = AsyncMock()
        if isinstance(command, GetAssetStreamCommand):
            mock_stream.get_first_non_none_value.return_value = lambda: create_dummy_iso_with_sfo()
        else:
            mock_stream.get_first_non_none_value.return_value = None
        return mock_stream

    mock_world.dispatch_command.side_effect = dispatch_command_side_effect_for_stream

    # Call command handler for each dispatched command
    for command in extract_commands:
        await psp_iso_metadata_extraction_command_handler_system(command, mock_transaction, mock_world)

    # 5. Assert command handler added components correctly
    assert mock_transaction.add_or_update_component.call_count == 4  # 2 for SFO, 2 for Raw SFO

    call_args_list = mock_transaction.add_or_update_component.call_args_list
    standalone_iso_calls = [c for c in call_args_list if c.args[0] == standalone_iso_entity_id]
    assert len(standalone_iso_calls) == 2
    assert isinstance(standalone_iso_calls[0].args[1], PSPSFOMetadataComponent)
    assert standalone_iso_calls[0].args[1].title == "Test Game"

    archived_iso_calls = [c for c in call_args_list if c.args[0] == archived_iso_entity_id]
    assert len(archived_iso_calls) == 2
    assert isinstance(archived_iso_calls[0].args[1], PSPSFOMetadataComponent)
    assert archived_iso_calls[0].args[1].title == "Test Game"
