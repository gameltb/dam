import io
from contextlib import redirect_stderr
from typing import Any

import pytest

from domarkx.utils.code_execution import execute_code_block


def test_execute_code_block_success() -> None:
    """Tests that execute_code_block successfully executes valid code."""
    local_vars: dict[str, Any] = {}
    execute_code_block("a = 1 + 2", local_vars=local_vars)
    assert local_vars["a"] == 3


def test_execute_code_block_failure_with_source() -> None:
    """Tests that execute_code_block prints a traceback with source code on failure."""
    code = "a = 1\nb = a / 0"
    f = io.StringIO()
    with redirect_stderr(f), pytest.raises(ZeroDivisionError):
        execute_code_block(code)

    s = f.getvalue()
    assert "b = a / 0" in s
