import logging
import os
from typing import List


def tool_list_files(path: str = ".") -> List[str]:
    """
    Lists all files and directories in the specified path.

    Args:
        path (str): The path to list files from. Defaults to the current directory.

    Returns:
        List[str]: A list of files and directories.
    """
    logging.info(f"Listing files in: {path}")
    try:
        return os.listdir(path)
    except FileNotFoundError:
        return ["Error: Directory not found."]
    except Exception as e:
        return [f"An unexpected error occurred: {e}"]
