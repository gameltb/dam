import logging
import os
from domarkx.tools.tool_decorator import tool_handler


@tool_handler()
def write_to_file(path: str, content: str) -> str:
    """
    Write the full content to a file. If the file does not exist, it will be created. If it exists, it will be overwritten.

    Args:
        path (str): Path to the file to write.
        content (str): Full content to write.

    Returns:
        str: Result message of the write operation.

    Raises:
        TypeError: If argument types are incorrect.
        OSError: If directory creation or file writing fails.
        PermissionError: If lacking permission to write file.
        IOError: If file write error occurs.
        Exception: For other unexpected errors.
    """
    logging.info(f"Attempting to write content to file: '{path}'.")

    if not isinstance(path, str):
        raise TypeError(f"Argument 'path' must be a string, but received {type(path).__name__}.")
    if not isinstance(content, str):
        raise TypeError(f"Argument 'content' must be a string, but received {type(content).__name__}.")

    actual_line_count = len(content.splitlines())

    try:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            logging.info(f"Creating directory: '{directory}'.")
            os.makedirs(directory)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        logging.info(f"File '{path}' written successfully, {actual_line_count} lines.")
        return f"File '{path}' written successfully, {actual_line_count} lines."
    except PermissionError as e:
        error_msg = f"No permission to write file '{path}': {e}"
        logging.error(error_msg)
        raise PermissionError(error_msg)
    except IOError as e:
        error_msg = f"IO error occurred while writing file '{path}': {e}"
        logging.error(error_msg)
        raise IOError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error occurred while writing file '{path}': {e}"
        logging.error(error_msg)
        raise Exception(error_msg)
