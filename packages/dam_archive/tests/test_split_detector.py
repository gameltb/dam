import pytest

from dam_archive.split_detector import SplitArchiveInfo, detect


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # WinRAR-style
        ("MyArchive.part1.rar", SplitArchiveInfo(base_name="MyArchive", part_num=1)),
        ("MyArchive.part01.rar", SplitArchiveInfo(base_name="MyArchive", part_num=1)),
        ("My.Archive.part2.rar", SplitArchiveInfo(base_name="My.Archive", part_num=2)),
        # 7-Zip style
        ("Another.Archive.7z.001", SplitArchiveInfo(base_name="Another.Archive", part_num=1)),
        ("Archive.7z.123", SplitArchiveInfo(base_name="Archive", part_num=123)),
        # Old RAR-style
        ("OldSchool.r00", SplitArchiveInfo(base_name="OldSchool", part_num=0)),
        ("OldSchool.r01", SplitArchiveInfo(base_name="OldSchool", part_num=1)),
        # ZIP style
        ("ZipArchive.z01", SplitArchiveInfo(base_name="ZipArchive", part_num=1)),
        ("ZipArchive.z15", SplitArchiveInfo(base_name="ZipArchive", part_num=15)),
        # Negative cases
        ("MyArchive.txt", None),
        ("MyArchive.part1.txt", None),
        ("MyArchive.rar", None),
        ("MyArchive.7z", None),
        ("MyArchive.part.rar", None),
        ("MyArchive.r.01", None),
    ],
)
def test_detect_split_archive(filename: str, expected: SplitArchiveInfo | None):
    """Tests that the split archive detection works for various patterns."""
    assert detect(filename) == expected
