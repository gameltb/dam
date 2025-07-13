import logging
import os


def list_files(path: str, recursive: bool = False) -> str:
    """
    列出指定目录下的文件和目录，返回它们的完整路径。

    参数:
        path (str): 要列出内容的目录路径。
        recursive (bool): 是否递归列出。

    返回:
        str: 文件和目录的完整路径列表，每行一个路径。

    抛出:
        TypeError: 如果 'path' 或 'recursive' 参数类型不正确。
        FileNotFoundError: 如果路径不存在。
        NotADirectoryError: 如果路径不是一个目录。
        PermissionError: 如果没有权限访问目录。
        Exception: 捕获其他未预料的错误。
    """
    if isinstance(recursive, str):
        recursive = recursive == "true"
    logging.info(f"尝试列出 '{path}' 的文件和目录。递归模式: {recursive}。")

    # 参数类型检查
    if not isinstance(path, str):
        error_msg = f"参数 'path' 必须是字符串类型，但接收到 {type(path).__name__}。"
        logging.error(error_msg)
        raise TypeError(error_msg)
    if not isinstance(recursive, bool):
        error_msg = f"参数 'recursive' 必须是布尔类型，但接收到 {type(recursive).__name__}。"
        logging.error(error_msg)
        raise TypeError(error_msg)

    if not os.path.exists(path):
        logging.error(f"路径 '{path}' 不存在。")
        raise FileNotFoundError(f"路径 '{path}' 不存在。")
    if not os.path.isdir(path):
        logging.error(f"路径 '{path}' 不是一个目录。")
        raise NotADirectoryError(f"路径 '{path}' 不是一个目录。")

    listed_paths = []

    try:
        if recursive:
            logging.info(f"递归列出目录 '{path}'。")
            for root, dirs, files in os.walk(path):
                # 过滤掉 __pycache__ 目录，并修改 dirs 列表以避免 os.walk 访问这些目录
                dirs[:] = [d for d in dirs if d not in ["__pycache__"]]

                for d in sorted(dirs):
                    listed_paths.append(os.path.join(root, d) + os.sep)  # 添加目录斜杠
                for f in sorted(files):
                    listed_paths.append(os.path.join(root, f))
        else:
            logging.info(f"非递归列出目录 '{path}'。")
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
        error_msg = f"没有权限访问目录 '{path}': {e}"
        logging.error(error_msg)
        raise PermissionError(error_msg)
    except Exception as e:
        error_msg = f"列出目录 '{path}' 时发生意外错误: {e}"
        logging.error(error_msg)
        raise Exception(error_msg)

    if not listed_paths:
        return f"在 '{path}' 中没有找到任何非隐藏文件或目录。"

    # 将路径按字母顺序排序并连接成字符串
    output_str = "\n".join(sorted(listed_paths))
    logging.info(f"成功列出 '{path}' 的内容。")
    return output_str
