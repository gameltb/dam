import logging
import os


def list_files(path: str, recursive: bool = False) -> str:
    """
    List files and directories under the specified path, returning their full paths.

    Args:
        path (str): Directory path to list.
        recursive (bool): Whether to list recursively.

    Returns:
        str: List of file and directory paths, one per line.

    Raises:
        TypeError: If 'path' or 'recursive' argument types are incorrect.
        FileNotFoundError: If path does not exist.
        NotADirectoryError: If path is not a directory.
        PermissionError: If lacking permission to access directory.
        Exception: For other unexpected errors.
    """
    logging.info(f"Attempting to list files and directories in '{path}'. Recursive: {recursive}.")

    # 参数类型检查
    if not isinstance(path, str):
        error_msg = f"Argument 'path' must be a string, but received {type(path).__name__}."
        logging.error(error_msg)
        raise TypeError(error_msg)
    if not isinstance(recursive, bool):
        error_msg = f"Argument 'recursive' must be a boolean, but received {type(recursive).__name__}."
        logging.error(error_msg)
        raise TypeError(error_msg)

    if not os.path.exists(path):
        logging.error(f"Path '{path}' does not exist.")
        raise FileNotFoundError(f"Path '{path}' does not exist.")
    if not os.path.isdir(path):
        logging.error(f"Path '{path}' is not a directory.")
        raise NotADirectoryError(f"Path '{path}' is not a directory.")

    listed_paths = []

    try:
        if recursive:
            logging.info(f"Recursively listing directory '{path}'.")
            for root, dirs, files in os.walk(path):
                # 过滤掉 __pycache__ 目录，并修改 dirs 列表以避免 os.walk 访问这些目录
                dirs[:] = [d for d in dirs if d not in ["__pycache__"]]

                for d in sorted(dirs):
                    listed_paths.append(os.path.join(root, d) + os.sep)  # 添加目录斜杠
                for f in sorted(files):
                    listed_paths.append(os.path.join(root, f))
        else:
            logging.info(f"Non-recursive listing of directory '{path}'.")
            entries = os.listdir(path)
            # 过滤掉 __pycache__ 目录
            entries = [e for e in entries if e != "__pycache__"]

            for entry in sorted(entries):
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    listed_paths.append(full_path + os.sep)  # 添加目录斜杠
                else:
                    listed_paths.append(full_path)
    except PermissionError as e:
        error_msg = f"No permission to access directory '{path}': {e}"
        logging.error(error_msg)
        raise PermissionError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error occurred while listing directory '{path}': {e}"
        logging.error(error_msg)
        raise Exception(error_msg)

    if not listed_paths:
        return f"No non-hidden files or directories found in '{path}'."

    # 将路径按字母顺序排序并连接成字符串
    output_str = "\n".join(sorted(listed_paths))
    logging.info(f"Successfully listed contents of '{path}'.")
    return output_str
