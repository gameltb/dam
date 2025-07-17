from domarkx.tools.tool_decorator import tool_handler
from domarkx.tools.unconstrained.execute_command import execute_command as execute_command_unconstrained


@tool_handler()
def execute_command(*args, **kwargs):
    return execute_command_unconstrained(*args, **kwargs)
