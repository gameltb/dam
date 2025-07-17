from domarkx.tools.tool_decorator import tool_handler
from domarkx.tools.unconstrained.read_file import read_file as read_file_unconstrained

read_file = tool_handler()(read_file_unconstrained)
