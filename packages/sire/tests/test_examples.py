import runpy

import pytest


def test_simple_usage_example():
    """
    Runs the simple_usage.py example to ensure it executes without errors.
    """
    try:
        runpy.run_path("examples/simple_usage.py", run_name="__main__")
    except Exception as e:
        pytest.fail(f"Running examples/simple_usage.py failed with {e}")
