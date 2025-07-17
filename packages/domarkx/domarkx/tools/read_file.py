from domarkx.tools.tool_decorator import tool_handler
from domarkx.tools.unconstrained.read_file import read_file as read_file_unconstrained


@tool_handler()
def read_file(*args, **kwargs):
    return read_file_unconstrained(*args, **kwargs)
