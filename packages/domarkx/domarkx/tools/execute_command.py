from domarkx.tools.tool_decorator import tool_handler
from domarkx.tools.unconstrained.execute_command import execute_command as execute_command_unconstrained

execute_command = tool_handler()(execute_command_unconstrained)
