"""A tool for writing content to a file."""

import logging
import pathlib

from domarkx.tools.tool_factory import tool_handler

logger = logging.getLogger(__name__)


@tool_handler()
def tool_write_to_file(path: str, content: str, append: bool = False) -> str:
    """
    Write content to the specified file path.

    Args:
        path (str): File path to write to.
        content (str): Content to write to the file.
        append (bool): If True, appends to the file instead of overwriting. Defaults to False.

    Returns:
        str: A confirmation message.

    Raises:
        Exception: For any unexpected errors.

    """
    mode = "a" if append else "w"
    logger.info("Writing to file: %s (mode: %s)", path, mode)
    try:
        # Create parent directories if they don't exist
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with p.open(mode, encoding="utf-8") as f:
            f.write(content)
        return f"File '{path}' written successfully."
    except Exception as e:
        logger.error("An unexpected error occurred while writing to file %s: %s", path, e)
        raise e
