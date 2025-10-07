"""A tool for executing shell commands."""

import logging
import subprocess

from domarkx.tools.tool_factory import tool_handler

logger = logging.getLogger(__name__)


@tool_handler()
def tool_execute_command(command: str) -> str:
    """
    Execute a shell command and return its output.

    Args:
        command (str): The command to execute.

    Returns:
        str: The output of the command.

    Raises:
        subprocess.CalledProcessError: If the command fails.

    """
    logger.info("Executing command: %s", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error("Error executing command: %s\n%s", e, e.stderr)
        raise e
