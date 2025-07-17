from domarkx.tools.tool_decorator import tool_handler
from domarkx.tools.unconstrained.write_to_file import write_to_file as write_to_file_unconstrained

write_to_file = tool_handler()(write_to_file_unconstrained)
