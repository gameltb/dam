from io import BytesIO
from unittest.mock import AsyncMock

import pycdlib
import pytest
from dam.core.world import World
from dam_app.commands import GetAssetStreamCommand
from dam_app.events import AssetsReadyForMetadataExtraction
from dam_app.models import ArchiveMemberComponent
from dam_fs.models import FilePropertiesComponent

from dam_psp import psp_iso_functions
from dam_psp.models import PSPSFOMetadataComponent
from dam_psp.systems import psp_iso_metadata_extraction_system

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
    iso = pycdlib.PyCdlib()
    iso.new(interchange_level=1, joliet=True)
    iso.add_directory("/PSP_GAME")
    iso.add_fp(BytesIO(DUMMY_SFO_CONTENT), len(DUMMY_SFO_CONTENT), "/PSP_GAME/PARAM.SFO;1")

    iso_fp = BytesIO()
    iso.write_fp(iso_fp)
    iso.close()

    iso_fp.seek(0)
    return iso_fp


@pytest.fixture
def dummy_iso_stream() -> BytesIO:
    """Pytest fixture to provide a dummy ISO stream."""
    return create_dummy_iso_with_sfo()


def test_process_iso_stream_extracts_sfo_metadata(dummy_iso_stream):
    """
    Tests that process_iso_stream correctly extracts metadata from a dummy ISO.
    """
    sfo = psp_iso_functions.process_iso_stream(dummy_iso_stream)

    assert sfo is not None
    sfo_metadata = sfo.data
    assert sfo_metadata is not None
    assert sfo_metadata.get("TITLE") == "Test Game"
    assert sfo_metadata.get("DISC_ID") == "ULUS-12345"


def test_process_iso_stream_handles_non_iso_file():
    """
    Tests that process_iso_stream raises an IOError for non-ISO files.
    """
    non_iso_stream = BytesIO(b"this is not an iso file")
    with pytest.raises(IOError):
        psp_iso_functions.process_iso_stream(non_iso_stream)


@pytest.mark.asyncio
async def test_psp_iso_metadata_extraction_system(mocker):
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

    async def get_component_side_effect(entity_id, component_type):
        if entity_id == standalone_iso_entity_id:
            if component_type == PSPSFOMetadataComponent:
                return None
            if component_type == ArchiveMemberComponent:
                return None
            if component_type == FilePropertiesComponent:
                return FilePropertiesComponent(original_filename="test.iso")
        elif entity_id == archived_iso_entity_id:
            if component_type == PSPSFOMetadataComponent:
                return None
            if component_type == ArchiveMemberComponent:
                return ArchiveMemberComponent(archive_entity_id=99, path_in_archive="game.iso")
        elif entity_id == non_iso_entity_id:
            if component_type == PSPSFOMetadataComponent:
                return None
            if component_type == ArchiveMemberComponent:
                return None
            if component_type == FilePropertiesComponent:
                return FilePropertiesComponent(original_filename="text.txt")
        return None

    mock_transaction.get_component.side_effect = get_component_side_effect
    mock_transaction.add_component_to_entity = AsyncMock()

    # Mock world
    mock_world = AsyncMock(spec=World)

    async def dispatch_command_side_effect(command):
        if isinstance(command, GetAssetStreamCommand):
            return create_dummy_iso_with_sfo()
        return None

    mock_world.dispatch_command.side_effect = dispatch_command_side_effect

    # Create event
    event = AssetsReadyForMetadataExtraction(
        entity_ids=[standalone_iso_entity_id, archived_iso_entity_id, non_iso_entity_id]
    )

    # 2. Execute
    await psp_iso_metadata_extraction_system(event, mock_transaction, mock_world)

    # 3. Assert
    # Check that add_component_to_entity was called for the two ISO entities
    assert mock_transaction.add_component_to_entity.call_count == 4  # 2 for SFO, 2 for Raw SFO

    # Check call for standalone ISO
    call_args_list = mock_transaction.add_component_to_entity.call_args_list
    standalone_iso_calls = [c for c in call_args_list if c.args[0] == standalone_iso_entity_id]
    assert len(standalone_iso_calls) == 2
    assert isinstance(standalone_iso_calls[0].args[1], PSPSFOMetadataComponent)
    assert standalone_iso_calls[0].args[1].title == "Test Game"

    # Check call for archived ISO
    archived_iso_calls = [c for c in call_args_list if c.args[0] == archived_iso_entity_id]
    assert len(archived_iso_calls) == 2
    assert isinstance(archived_iso_calls[0].args[1], PSPSFOMetadataComponent)
    assert archived_iso_calls[0].args[1].title == "Test Game"

    # Check that dispatch_command was called for the two ISO entities
    assert mock_world.dispatch_command.call_count == 2
    dispatch_calls = mock_world.dispatch_command.call_args_list
    assert dispatch_calls[0].args[0].entity_id == standalone_iso_entity_id
    assert dispatch_calls[1].args[0].entity_id == archived_iso_entity_id
