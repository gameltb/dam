import os
import pathlib
import shutil

import pytest

from domarkx.action.exec_doc import aexec_doc


def get_function_execution_tests_files():
    function_execution_tests_dir = os.path.join(os.path.dirname(__file__), "function_execution_tests")
    if not os.path.isdir(function_execution_tests_dir):
        return []
    return [
        os.path.join(function_execution_tests_dir, f)
        for f in os.listdir(function_execution_tests_dir)
        if f.endswith(".md")
    ]


@pytest.mark.parametrize("filepath", get_function_execution_tests_files())
@pytest.mark.asyncio
async def test_function_execution_in_temp_dir(filepath, tmp_path):
    """
    Tests that function_execution_test markdown files conform to the domarkx documentation format.
    """
    temp_doc_path = tmp_path / os.path.basename(filepath)
    shutil.copy(filepath, temp_doc_path)
    doc_path = pathlib.Path(temp_doc_path)
    await aexec_doc(doc_path, handle_one_toolcall=True, allow_user_message_in_FunctionExecution=False)
