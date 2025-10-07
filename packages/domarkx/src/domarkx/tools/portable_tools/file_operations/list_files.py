import logging
import pathlib


def tool_list_files(path: str = ".") -> list[str]:
    """
    Lists all files and directories in the specified path.

    Args:
        path (str): The path to list files from. Defaults to the current directory.

    Returns:
        list[str]: A list of files and directories.

    Raises:
        FileNotFoundError: If the directory is not found.
        Exception: For other unexpected errors.

    """
    logging.info(f"Listing files in: {path}")
    try:
        return [p.name for p in pathlib.Path(path).iterdir()]
    except FileNotFoundError as e:
        logging.error(f"Error listing files in {path}: {e}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred while listing files in {path}: {e}")
        raise e
