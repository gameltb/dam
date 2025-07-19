import logging
import os


def tool_write_to_file(path: str, content: str, append: bool = False) -> str:
    """
    Writes content to the specified file path.

    Args:
        path (str): File path to write to.
        content (str): Content to write to the file.
        append (bool): If True, appends to the file instead of overwriting. Defaults to False.

    Returns:
        str: A confirmation message.
    """
    mode = "a" if append else "w"
    logging.info(f"Writing to file: {path} (mode: {mode})")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        return f"File '{path}' written successfully."
    except Exception as e:
        return f"An unexpected error occurred: {e}"
