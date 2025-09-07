import asyncio
import pathlib
import re
import time
from typing import Any, Generator
from unittest.mock import patch

import pytest

from domarkx.action.exec_doc import aexec_doc

VALID_MD_CONTENT = "## user\n\n> hello"


class StopTest(Exception):
    pass


@pytest.fixture
def stop_after_file_creation() -> Generator[Any, None, None]:
    with patch("domarkx.action.exec_doc.AutoGenSession", side_effect=StopTest) as mock_session:
        yield mock_session


def test_exec_doc_creates_timestamped_file(tmp_path: pathlib.Path, stop_after_file_creation: Any) -> None:
    test_file = tmp_path / "test.md"
    test_file.write_text(VALID_MD_CONTENT)

    with pytest.raises(StopTest):
        asyncio.run(aexec_doc(test_file, overwrite=False))

    created_files = list(tmp_path.glob("test_*.md"))
    assert len(created_files) == 1
    new_file = created_files[0]
    assert re.match(r"test_\d{8}_\d{6}\.md", new_file.name)


def test_exec_doc_appends_A_to_timestamped_file(tmp_path: pathlib.Path, stop_after_file_creation: Any) -> None:
    ts_file = tmp_path / "test_20250101_120000.md"
    ts_file.write_text(VALID_MD_CONTENT)

    with pytest.raises(StopTest):
        asyncio.run(aexec_doc(ts_file, overwrite=False))

    new_file = tmp_path / "test_20250101_120000A.md"
    assert new_file.exists()


def test_exec_doc_creates_new_timestamp_if_A_exists(tmp_path: pathlib.Path, stop_after_file_creation: Any) -> None:
    ts_file = tmp_path / "test_20250101_120000.md"
    ts_file.write_text(VALID_MD_CONTENT)

    ts_A_file = tmp_path / "test_20250101_120000A.md"
    ts_A_file.write_text(VALID_MD_CONTENT)

    # To ensure a new timestamp is generated
    time.sleep(1)

    with pytest.raises(StopTest):
        asyncio.run(aexec_doc(ts_file, overwrite=False))

    created_files = list(tmp_path.glob("test_*.md"))
    # We should have 3 files: original, original_A, and the new one with a new timestamp
    assert len(created_files) == 3

    new_file = None
    for f in created_files:
        if f.name not in [ts_file.name, ts_A_file.name]:
            new_file = f
            break

    assert new_file is not None
    assert re.match(r"test_\d{8}_\d{6}\.md", new_file.name)
    assert new_file.name != ts_file.name
