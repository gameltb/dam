import hashlib
import zlib
from io import BytesIO

import pycdlib
import pytest

from dam.services import psp_iso_service

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
    result = psp_iso_service.process_iso_stream(dummy_iso_stream)

    assert result is not None
    sfo_metadata = result.get("sfo_metadata")
    assert sfo_metadata is not None
    assert sfo_metadata.get("TITLE") == "Test Game"
    assert sfo_metadata.get("DISC_ID") == "ULUS-12345"


def test_process_iso_stream_extracts_raw_sfo_metadata(dummy_iso_stream):
    """
    Tests that process_iso_stream correctly extracts raw metadata from a dummy ISO.
    """
    result = psp_iso_service.process_iso_stream(dummy_iso_stream)

    assert result is not None
    sfo_raw_metadata = result.get("sfo_raw_metadata")
    assert sfo_raw_metadata is not None
    assert sfo_raw_metadata.get("TITLE") == "Test Game"
    assert sfo_raw_metadata.get("DISC_ID") == "ULUS-12345"


def test_process_iso_stream_calculates_hashes(dummy_iso_stream):
    """
    Tests that process_iso_stream correctly calculates all required hashes.
    """
    iso_content = dummy_iso_stream.read()
    dummy_iso_stream.seek(0)  # Reset for the function to read

    result = psp_iso_service.process_iso_stream(dummy_iso_stream)

    assert result is not None
    hashes = result.get("hashes")
    assert hashes is not None

    # Pre-calculate expected hashes
    expected_md5 = hashlib.md5(iso_content).digest()
    expected_sha1 = hashlib.sha1(iso_content).digest()
    expected_sha256 = hashlib.sha256(iso_content).digest()
    expected_crc32 = zlib.crc32(iso_content).to_bytes(4, "big")

    assert hashes.get("md5") == expected_md5
    assert hashes.get("sha1") == expected_sha1
    assert hashes.get("sha256") == expected_sha256
    assert hashes.get("crc32") == expected_crc32


def test_process_iso_stream_handles_non_iso_file():
    """
    Tests that process_iso_stream raises an IOError for non-ISO files.
    """
    non_iso_stream = BytesIO(b"this is not an iso file")
    with pytest.raises(IOError):
        psp_iso_service.process_iso_stream(non_iso_stream)


# Tests for the ingestion system
from unittest.mock import AsyncMock, MagicMock

from dam.systems import psp_iso_ingestion_system


@pytest.mark.asyncio
async def test_ingest_single_iso_file(tmp_path, mocker):
    """
    Tests that the ingestion system can process a single, standalone ISO file.
    """
    # 1. Setup
    # Create a dummy ISO file in the temporary directory
    iso_path = tmp_path / "test.iso"
    dummy_iso_content = create_dummy_iso_with_sfo().read()
    iso_path.write_bytes(dummy_iso_content)

    # Mock the service and database interactions
    mock_process_iso_stream = mocker.patch(
        "dam.services.psp_iso_service.process_iso_stream",
        return_value={
            "hashes": {"md5": b"md5_hash", "sha1": b"sha1_hash", "sha256": b"sha256_hash", "crc32": b"crc32_hash"},
            "sfo_metadata": {"TITLE": "Test Game", "DISC_ID": "ULUS-12345"},
            "sfo_raw_metadata": {"TITLE": "Test Game", "DISC_ID": "ULUS-12345"},
        },
    )

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    mocker.patch("dam.services.ecs_service.create_entity", new_callable=AsyncMock, return_value=MagicMock(id=1))
    mock_add_component = mocker.patch("dam.services.ecs_service.add_component_to_entity", new_callable=AsyncMock)

    # 2. Execute
    await psp_iso_ingestion_system.ingest_psp_isos_from_directory(session=mock_session, directory=str(tmp_path))

    # 3. Assert
    mock_process_iso_stream.assert_called_once()

    # Check that a new entity was created
    from dam.services.ecs_service import create_entity

    create_entity.assert_awaited_once_with(mock_session)

    # Check that all components were added
    assert mock_add_component.await_count == 6  # 4 hashes + 1 SFO + 1 raw SFO

    # More detailed check of component data could be done here if needed
    # e.g. by inspecting the `call_args_list` of `mock_add_component`


@pytest.mark.asyncio
async def test_ingest_skips_duplicate_iso_file(tmp_path, mocker):
    """
    Tests that the ingestion system skips a file if its hash already exists.
    """
    # 1. Setup
    iso_path = tmp_path / "duplicate.iso"
    dummy_iso_content = create_dummy_iso_with_sfo().read()
    iso_path.write_bytes(dummy_iso_content)

    mocker.patch(
        "dam.services.psp_iso_service.process_iso_stream",
        return_value={
            "hashes": {"md5": b"duplicate_md5_hash"},
            "sfo_metadata": {"TITLE": "Duplicate Game"},
        },
    )

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = 1  # Return a dummy entity ID
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_create_entity = mocker.patch("dam.services.ecs_service.create_entity", new_callable=AsyncMock)
    mock_add_component = mocker.patch("dam.services.ecs_service.add_component_to_entity", new_callable=AsyncMock)

    # 2. Execute
    await psp_iso_ingestion_system.ingest_psp_isos_from_directory(session=mock_session, directory=str(tmp_path))

    # 3. Assert
    # Ensure entity and components were NOT created
    mock_create_entity.assert_not_awaited()
    mock_add_component.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_iso_from_7z_file(tmp_path, mocker):
    """
    Tests that the system can find and process an ISO file inside a zip archive.
    """
    # 1. Setup
    zip_path = tmp_path / "archive.zip"
    dummy_iso_content = create_dummy_iso_with_sfo().read()

    import zipfile

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test.iso", dummy_iso_content)

    mock_process_iso_stream = mocker.patch(
        "dam.services.psp_iso_service.process_iso_stream",
        return_value={
            "hashes": {"md5": b"some_hash", "sha1": b"sha1_hash", "sha256": b"sha256_hash", "crc32": b"crc32_hash"},
            "sfo_metadata": {},
        },
    )
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mocker.patch("dam.services.ecs_service.create_entity", new_callable=AsyncMock)
    mock_add_component = mocker.patch("dam.services.ecs_service.add_component_to_entity", new_callable=AsyncMock)

    # 2. Execute
    await psp_iso_ingestion_system.ingest_psp_isos_from_directory(session=mock_session, directory=str(tmp_path))

    # 3. Assert
    mock_process_iso_stream.assert_called_once()
    from dam.services.ecs_service import create_entity

    create_entity.assert_awaited_once()
    assert mock_add_component.await_count > 0
