from pathlib import Path

import pytest

from dam_archive.base import to_stream_provider
from dam_archive.exceptions import InvalidPasswordError
from dam_archive.handlers.sevenzip_cli import SevenZipCliArchiveHandler


@pytest.mark.asyncio
async def test_open_regular_archive(regular_7z_archive: Path):
    handler = await SevenZipCliArchiveHandler.create(to_stream_provider(str(regular_7z_archive)))
    assert len(handler.list_files()) == 1
    assert handler.list_files()[0].name == "file.txt"
    await handler.close()


@pytest.mark.asyncio
async def test_open_protected_archive_with_password(protected_7z_archive: Path):
    handler = await SevenZipCliArchiveHandler.create(to_stream_provider(str(protected_7z_archive)), password="password")
    assert len(handler.list_files()) == 1
    assert handler.list_files()[0].name == "file.txt"
    await handler.close()


@pytest.mark.asyncio
async def test_open_protected_archive_with_wrong_password(protected_7z_archive: Path):
    with pytest.raises(InvalidPasswordError):
        await SevenZipCliArchiveHandler.create(to_stream_provider(str(protected_7z_archive)), password="wrong_password")


@pytest.mark.asyncio
async def test_open_bcj2_archive(bcj2_7z_archive: Path):
    handler = await SevenZipCliArchiveHandler.create(to_stream_provider(str(bcj2_7z_archive)))
    assert len(handler.list_files()) == 1
    assert handler.list_files()[0].name == "hello_x86"
    await handler.close()


@pytest.mark.asyncio
async def test_iter_files(regular_7z_archive: Path):
    handler = await SevenZipCliArchiveHandler.create(to_stream_provider(str(regular_7z_archive)))
    files = list(handler.iter_files())
    assert len(files) == 1
    member_info, file_handle = files[0]
    assert member_info.name == "file.txt"
    content = file_handle.read()
    assert content == b"This is a test file for 7z archives.\n"
    file_handle.close()
    await handler.close()


@pytest.mark.asyncio
async def test_open_file(regular_7z_archive: Path):
    handler = await SevenZipCliArchiveHandler.create(to_stream_provider(str(regular_7z_archive)))
    member_info, file_handle = handler.open_file("file.txt")
    assert member_info.name == "file.txt"
    content = file_handle.read()
    assert content == b"This is a test file for 7z archives.\n"
    file_handle.close()
    await handler.close()
