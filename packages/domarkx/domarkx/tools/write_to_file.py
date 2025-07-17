from domarkx.tools.tool_decorator import tool_handler
from domarkx.tools.unconstrained.write_to_file import write_to_file as write_to_file_unconstrained


@tool_handler()
def write_to_file(*args, **kwargs):
    return write_to_file_unconstrained(*args, **kwargs)
