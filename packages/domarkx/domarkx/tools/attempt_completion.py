import logging

from domarkx.tools.tool_factory import _tool_handler as tool_handler


@tool_handler()
def attempt_completion_tool(result: str, command: str = None) -> str:
    """
    Indicates that a task is complete, providing the result and an optional display command.

    Args:
        result (str): Description of the final result of the task.
        command (str): CLI command to display the result.

    Returns:
        str: Formatted completion information string.

    Raises:
        TypeError: If 'result' or 'command' argument types are incorrect.
    """
    logging.info("Attempting to format task completion information.")

    if not isinstance(result, str):
        error_msg = f"Argument 'result' must be a string, but received {type(result).__name__}."
        logging.error(error_msg)
        raise TypeError(error_msg)

    if command is not None and not isinstance(command, str):
        error_msg = f"Argument 'command' must be a string or None, but received {type(command).__name__}."
        logging.error(error_msg)
        raise TypeError(error_msg)

    command_tag = f"<command>{command}</command>\n" if command else ""

    formatted_completion = f"<attempt_completion>\n<result>\n{result}\n</result>\n{command_tag}</attempt_completion>"
    logging.info("Successfully formatted task completion information.")
    return formatted_completion
