import zipfile
from pathlib import Path

import pytest

from dam_archive.main import open_archive

FILE_CONTENT = b"content"


@pytest.fixture
def utf8_encoded_zip_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "utf8_test.zip"
    # "测试" means "test" in Chinese.
    filename_in_zip = "测试.txt"
    with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename_in_zip, FILE_CONTENT)
    return file_path


@pytest.mark.asyncio
async def test_open_zip_with_utf8_filename(utf8_encoded_zip_file: Path) -> None:
    # This test checks the happy path where the filename is UTF-8 encoded
    # and the UTF-8 flag is set in the zip file.
    archive = await open_archive(utf8_encoded_zip_file, "application/zip")
    assert archive is not None
    files = archive.list_files()
    assert "测试.txt" in [m.name for m in files]

    member_info, f_in_zip = archive.open_file("测试.txt")
    with f_in_zip:
        assert f_in_zip.read() == FILE_CONTENT
    assert member_info.name == "测试.txt"


@pytest.fixture
def cp437_encoded_zip_file(tmp_path: Path) -> Path:
    # It's difficult to create a zip file with a specific non-UTF-8 encoding
    # for the filename using the standard zipfile library.
    # This test creates a zip file where the filename is encoded with cp437,
    # which is the default for zipfile when the UTF-8 flag is not set.
    file_path = tmp_path / "cp437_test.zip"
    filename_in_zip = "tést.txt"  # é is in cp437
    with zipfile.ZipFile(file_path, "w") as zf:
        # We need to encode the filename to cp437 bytes ourselves
        # and use a ZipInfo object to bypass the default UTF-8 encoding.
        zinfo = zipfile.ZipInfo(filename_in_zip)
        zinfo.flag_bits &= ~0x800  # Unset the UTF-8 flag
        zf.writestr(zinfo, FILE_CONTENT)
    return file_path


@pytest.mark.asyncio
async def test_open_zip_with_cp437_filename(cp437_encoded_zip_file: Path) -> None:
    # This test checks the fallback path where the filename is not UTF-8.
    # My code attempts to decode with cp437, then utf-8, then gbk.
    # In this case, the filename is valid cp437.
    archive = await open_archive(cp437_encoded_zip_file, "application/zip")
    assert archive is not None
    files = archive.list_files()
    assert "tést.txt" in [m.name for m in files]


@pytest.mark.asyncio
async def test_iter_files_zip_with_utf8_filename(utf8_encoded_zip_file: Path) -> None:
    # This test checks the iter_files method.
    archive = await open_archive(utf8_encoded_zip_file, "application/zip")
    assert archive is not None

    files = list(archive.iter_files())
    assert len(files) == 1

    member_info, member_stream = files[0]
    assert member_info.name == "测试.txt"
    assert member_info.size == len(FILE_CONTENT)

    with member_stream as f_in_zip:
        assert f_in_zip.read() == FILE_CONTENT
