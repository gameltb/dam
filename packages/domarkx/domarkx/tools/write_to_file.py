import logging
import os


def write_to_file(path: str, content: str) -> str:
    """
    将完整内容写入文件。如果文件不存在，则创建。如果存在，则覆盖。

    参数:
        path (str): 要写入的文件的路径。
        content (str): 要写入的完整内容。

    返回:
        str: 写入操作的结果信息。

    抛出:
        TypeError: 如果参数类型不正确。
        OSError: 如果目录创建或文件写入失败。
        PermissionError: 如果没有权限写入文件。
        IOError: 如果文件写入发生错误。
        Exception: 捕获其他未预料的错误。
    """
    logging.info(f"尝试将内容写入文件: '{path}'。")

    # 参数类型检查
    if not isinstance(path, str):
        raise TypeError(f"参数 'path' 必须是字符串类型，但接收到 {type(path).__name__}。")
    if not isinstance(content, str):
        raise TypeError(f"参数 'content' 必须是字符串类型，但接收到 {type(content).__name__}。")

    # 简化行数计算，仅用于日志输出
    actual_line_count = len(content.splitlines())

    try:
        # 确保目录存在
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            logging.info(f"创建目录: '{directory}'。")
            os.makedirs(directory)

        # 写入文件
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        logging.info(f"文件 '{path}' 已成功写入，共 {actual_line_count} 行。")
        return f"文件 '{path}' 已成功写入，共 {actual_line_count} 行。"
    except PermissionError as e:
        error_msg = f"没有权限写入文件 '{path}': {e}"
        logging.error(error_msg)
        raise PermissionError(error_msg)
    except IOError as e:
        error_msg = f"写入文件 '{path}' 时发生 IO 错误: {e}"
        logging.error(error_msg)
        raise IOError(error_msg)
    except Exception as e:
        error_msg = f"写入文件 '{path}' 时发生意外错误: {e}"
        logging.error(error_msg)
        raise Exception(error_msg)
