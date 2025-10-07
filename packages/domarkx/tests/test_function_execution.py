"""Tests for the function execution documents."""

import pathlib
import shutil

import pytest

from domarkx.action.exec_doc import aexec_doc


def get_function_execution_tests_files() -> list[str]:
    """Get a list of all function execution test markdown files."""
    function_execution_tests_dir = pathlib.Path(__file__).parent / "function_execution_tests"
    if not function_execution_tests_dir.is_dir():
        return []
    return [str(p) for p in function_execution_tests_dir.glob("*.md")]


@pytest.mark.parametrize("filepath", get_function_execution_tests_files())
@pytest.mark.asyncio
async def test_function_execution_in_temp_dir(filepath: str, tmp_path: pathlib.Path) -> None:
    """Tests that function_execution_test markdown files conform to the domarkx documentation format."""
    temp_doc_path = tmp_path / pathlib.Path(filepath).name
    shutil.copy(filepath, temp_doc_path)
    doc_path = pathlib.Path(temp_doc_path)
    await aexec_doc(doc_path, handle_one_toolcall=True, allow_user_message_in_function_execution=False)
