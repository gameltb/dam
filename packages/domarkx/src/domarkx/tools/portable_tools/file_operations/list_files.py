"""A tool for listing files in a directory."""

import logging
import pathlib

from domarkx.tools.tool_factory import tool_handler

logger = logging.getLogger(__name__)


@tool_handler()
def tool_list_files(path: str = ".") -> list[str]:
    """
    List all files and directories in the specified path.

    Args:
        path (str): The path to list files from. Defaults to the current directory.

    Returns:
        list[str]: A list of files and directories.

    Raises:
        FileNotFoundError: If the directory is not found.
        Exception: For other unexpected errors.

    """
    logger.info("Listing files in: %s", path)
    try:
        return [p.name for p in pathlib.Path(path).iterdir()]
    except FileNotFoundError as e:
        logger.error("Error listing files in %s: %s", path, e)
        raise e
    except Exception as e:
        logger.error("An unexpected error occurred while listing files in %s: %s", path, e)
        raise e
