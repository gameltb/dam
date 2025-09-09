import zipfile
from pathlib import Path

import pytest
from dam.core.world import World
from dam_fs.plugin import FsPlugin

from dam_archive.plugin import ArchivePlugin

pytest_plugins = ["dam_test_utils.fixtures"]


@pytest.fixture(scope="session")
def test_archives(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """
    Creates regular zip and password-protected 7z files for testing.
    """
    tmp_dir = tmp_path_factory.mktemp("archives")
    content_dir = tmp_dir / "content"
    content_dir.mkdir()
    (content_dir / "file1.txt").write_text("file one")
    (content_dir / "file2.txt").write_text("file two")

    # Create regular zip
    regular_zip_path = tmp_dir / "regular.zip"
    with zipfile.ZipFile(regular_zip_path, "w") as zf:
        for file in content_dir.iterdir():
            zf.write(file, arcname=file.name)

    # Create protected zip
    protected_zip_path = tmp_dir / "protected.zip"
    with zipfile.ZipFile(protected_zip_path, "w") as zf:
        zf.writestr("file1.txt", "file one", compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("file2.txt", "file two", compress_type=zipfile.ZIP_DEFLATED)
        zf.setpassword(b"password")

    return regular_zip_path, protected_zip_path


@pytest.fixture(autouse=True)
def setup_world_with_plugins(test_world_alpha: World) -> None:
    # The archive plugin depends on the fs plugin, so we need both.
    test_world_alpha.add_plugin(FsPlugin())
    test_world_alpha.add_plugin(ArchivePlugin())
