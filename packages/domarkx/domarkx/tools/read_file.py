import glob
import logging
import os
from typing import List, Optional, Union


def read_file(path: Union[str, List[str]], start_line: int = None, end_line: int = None) -> str:
    """
    读取指定路径的文件内容。适用于检查已知或未知内容的文件。
    可指定 `start_line` 和 `end_line` 高效读取部分部分内容。
    支持通配符 `*` 来一次性读取多个文件（例如 `*.py`, `dir/*.md`）。
    当使用通配符或传入文件列表时，`start_line` 和 `end_line` 参数将被忽略，将返回所有匹配/指定文件的完整内容。

    参数:
        path (Union[str, List[str]]): 要读取的文件路径，可以是单个路径字符串（可包含通配符），
                                     或一个文件路径列表。使用 `**` 来递归匹配目录。
        start_line (int): (可选) 开始行号（1-based）。
        end_line (int): (可选) 结束行号（1-based，包含）。

    返回:
        str: 文件的内容。如果匹配到多个文件或传入文件列表，则会显示每个文件的内容，并以文件名分隔。

    抛出:
        TypeError: 如果参数类型不正确。
        FileNotFoundError: 如果文件不存在。
        IsADirectoryError: 如果路径是一个目录。
        PermissionError: 如果没有权限读取文件。
        ValueError: 如果行号范围无效。
        IOError: 如果文件读写发生错误。
        Exception: 捕获其他未预料的错误。
    """
    start_line = int(start_line) if start_line else None
    end_line = int(end_line) if end_line else None
    logging.info(f"尝试读取文件: '{path}'。开始行: {start_line}, 结束行: {end_line}。")

    # 定义一个内部函数来处理单个文件的读取逻辑
    def _read_single_file(file_path: str, s_line: Optional[int], e_line: Optional[int]) -> str:
        logging.info(f"正在读取单个文件: '{file_path}'。")

        if not os.path.exists(file_path):
            logging.error(f"文件 '{file_path}' 不存在。")
            raise FileNotFoundError(f"文件 '{file_path}' 不存在。")
        if not os.path.isfile(file_path):
            logging.error(f"路径 '{file_path}' 是一个目录，不是文件。")
            raise IsADirectoryError(f"路径 '{file_path}' 是一个目录，不是文件。")

        # 模拟PDF和DOCX处理
        if file_path.lower().endswith((".pdf", ".docx")):
            logging.warning(f"注意: '{file_path}' 是二进制文件 (PDF/DOCX)。此工具模拟提取原始文本。")
            return f"注意: '{file_path}' 是二进制文件 (PDF/DOCX)。此工具模拟提取原始文本，实际实现会使用相应库。\n模拟内容: 这是从 {file_path} 提取的文本内容。"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            logging.info(f"成功读取文件 '{file_path}' 的 {len(lines)} 行。")

            output_lines = []
            # 调整为0-based索引
            start_idx = (s_line - 1) if s_line is not None and s_line > 0 else 0
            end_idx = e_line if e_line is not None and e_line > 0 else len(lines)

            # 确保索引在有效范围内
            start_idx = max(0, min(start_idx, len(lines)))
            end_idx = max(0, min(end_idx, len(lines)))

            if start_idx >= end_idx and s_line is not None and e_line is not None:
                error_msg = f"指定的行范围 (从 {s_line} 到 {e_line}) 无效或为空，因为文件只有 {len(lines)} 行。没有内容被读取。"
                logging.warning(error_msg)
                raise ValueError(error_msg)  # 抛出 ValueError 而不是返回字符串

            for i in range(start_idx, end_idx):
                output_lines.append(lines[i].rstrip("\n"))  # 移除行号前缀

            logging.info(f"成功从文件 '{file_path}' 提取指定行内容。")
            return "\n".join(output_lines)
        except PermissionError as e:
            error_msg = f"没有权限读取文件 '{file_path}': {e}"
            logging.error(error_msg)
            raise PermissionError(error_msg)
        except IOError as e:
            error_msg = f"读取文件 '{file_path}' 时发生 IO 错误: {e}"
            logging.error(error_msg)
            raise IOError(error_msg)
        except Exception as e:
            error_msg = f"读取文件 '{file_path}' 时发生意外错误: {e}"
            logging.error(error_msg)
            raise Exception(error_msg)

    # 统一处理多文件读取的逻辑
    def _read_multiple_files(file_list: List[str]) -> str:
        results = []
        for file_item in sorted(file_list):
            try:
                # 忽略 start_line 和 end_line for multiple files
                file_content = _read_single_file(file_item, None, None)
                results.append(f"--- 文件: {file_item} ---\n{file_content}\n")
            except (FileNotFoundError, IsADirectoryError, PermissionError, IOError, ValueError, Exception) as e:
                logging.error(f"处理文件 '{file_item}' 时发生错误: {e}")
                results.append(f"--- 文件: {file_item} ---\n错误: {e}\n")  # 记录错误但继续处理其他文件
        if not results:
            error_msg = "所有指定的文件都无法读取或不存在。"
            logging.error(error_msg)
            raise RuntimeError(error_msg)  # 如果所有文件都失败了，则抛出运行时错误
        return "\n".join(results)

    # 主逻辑分支
    if isinstance(path, list):
        logging.info(f"路径参数是列表，将处理 {len(path)} 个文件。")
        return _read_multiple_files(path)
    elif isinstance(path, str):
        # 检查路径是否包含通配符
        if glob.has_magic(path):
            logging.info(f"路径 '{path}' 包含通配符，将处理多个文件。")
            # 检查是否包含递归通配符，以便在 glob.glob 中设置 recursive=True
            use_recursive = "**" in path

            dirname, basename = os.path.split(path)
            directory_to_scan = dirname if dirname else "."

            full_glob_pattern = os.path.join(directory_to_scan, basename)
            matching_files = glob.glob(full_glob_pattern, recursive=use_recursive)
            logging.info(f"找到 {len(matching_files)} 个匹配 '{path}' 的文件 (递归: {use_recursive})。")

            if not matching_files:
                warning_msg = f"警告: 未找到与通配符模式 '{path}' 匹配的文件。"
                logging.warning(warning_msg)
                return warning_msg  # 仍然返回警告，而不是抛出错误，因为这不一定是失败

            return _read_multiple_files(matching_files)
        else:
            # 如果没有通配符，则回退到原来的单个文件读取逻辑
            return _read_single_file(path, start_line, end_line)
    else:
        error_msg = f"参数 'path' 必须是字符串或字符串列表，但接收到 {type(path).__name__}。"
        logging.error(error_msg)
        raise TypeError(error_msg)
