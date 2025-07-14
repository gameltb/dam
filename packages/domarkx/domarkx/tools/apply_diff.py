import logging
import os
import re


def apply_diff_tool(path: str, diff: str) -> str:
    """
    使用搜索和替换块来修改文件。

    参数:
        path (str): 要修改的文件的路径。
        diff (str): 定义更改的搜索/替换块。

    返回:
        str: 应用 diff 的结果信息。
    """
    logging.info(f"尝试对文件 '{path}' 应用 diff。")

    if not os.path.exists(path):
        logging.error(f"文件 '{path}' 不存在。")
        raise FileNotFoundError(f"文件 '{path}' 不存在。")
    if not os.path.isfile(path):
        logging.error(f"路径 '{path}' 是一个目录，不是文件。")
        raise IsADirectoryError(f"路径 '{path}' 是一个目录，不是文件。")

    try:
        with open(path, "r", encoding="utf-8") as f:
            original_lines = f.readlines()
        logging.info(f"成功读取文件 '{path}'。")
    except IOError as e:
        logging.error(f"读取文件 '{path}' 时发生 IO 错误: {e}")
        raise IOError(f"无法读取文件 '{path}': {e}")

    current_lines = list(original_lines)  # 创建一个副本以进行修改

    operations = []
    diff_lines = diff.splitlines(keepends=True)
    current_line_in_diff = 0

    logging.info("开始解析 diff 块。")
    while current_line_in_diff < len(diff_lines):
        line = diff_lines[current_line_in_diff]

        if re.fullmatch(r"\s*<<<<<<< ?SEARCH\s*", line.strip()):
            logging.info(f"发现 SEARCH 块，从 diff 行 {current_line_in_diff + 1} 开始解析。")
            current_line_in_diff += 1
            if current_line_in_diff >= len(diff_lines):
                raise ValueError("无效的 diff 格式: SEARCH 块后缺少内容。")

            line = diff_lines[current_line_in_diff].strip()
            if not line.startswith(":start_line:"):
                raise ValueError("无效的 diff 格式: SEARCH 块后缺少 ':start_line:'。")
            try:
                start_line_num = int(line.split(":")[2].strip())
            except (IndexError, ValueError):
                raise ValueError("无效的 diff 格式: ':start_line:' 后行号无效。")
            current_line_in_diff += 1

            if current_line_in_diff >= len(diff_lines) or not re.fullmatch(r"\s*-------\s*", diff_lines[current_line_in_diff].strip()):
                raise ValueError("无效的 diff 格式: SEARCH 块后缺少 '-------' 分隔符。")
            current_line_in_diff += 1

            search_content_lines = []
            while current_line_in_diff < len(diff_lines) and not re.fullmatch(r"\s*=======\s*", diff_lines[current_line_in_diff].strip()):
                search_content_lines.append(diff_lines[current_line_in_diff])
                current_line_in_diff += 1
            search_content = "".join(search_content_lines)

            if current_line_in_diff >= len(diff_lines) or not re.fullmatch(r"\s*=======\s*", diff_lines[current_line_in_diff].strip()):
                raise ValueError("无效的 diff 格式: SEARCH 内容后缺少 '=======' 分隔符。")
            current_line_in_diff += 1

            replace_content_lines = []
            while current_line_in_diff < len(diff_lines):
                line = diff_lines[current_line_in_diff]
                if re.fullmatch(r"\s*>{5,9} ?REPLACE\s*", line.strip()):
                    break  # Found end marker
                replace_content_lines.append(line)
                current_line_in_diff += 1
            replace_content = "".join(replace_content_lines)

            if current_line_in_diff >= len(diff_lines) or not re.fullmatch(r"\s*>{5,9} ?REPLACE\s*", diff_lines[current_line_in_diff].strip()):
                raise ValueError("无效的 diff 格式: REPLACE 内容后缺少 '>>>>>>> REPLACE' 结束标记。")
            current_line_in_diff += 1

            start_idx = start_line_num - 1
            # Use splitlines(keepends=True) to preserve original line endings for accurate comparison
            normalized_search_lines = search_content.splitlines(keepends=True)

            if start_idx < 0 or start_idx > len(original_lines):
                raise ValueError(f"Diff 块中的起始行号 {start_line_num} 超出文件 '{path}' 的界限 (1 到 {len(original_lines) + 1})。")

            search_window = 5
            found_match = False
            original_len = len(original_lines)
            search_len = len(normalized_search_lines)

            if search_len == 0 and search_content.strip() != "":
                raise ValueError("搜索内容无法解析为有效的行列表。")
            if search_len == 0 and search_content.strip() == "":
                found_match = True
            else:
                logging.info(f"在文件 '{path}' 中搜索匹配内容，从行 {start_line_num} +/- {search_window}。")
                for current_search_idx in range(max(0, start_idx - search_window), min(original_len - search_len + 1, start_idx + search_window + 1)):
                    actual_window_content = original_lines[current_search_idx : current_search_idx + search_len]

                    # Normalize both for comparison (strip leading/trailing whitespace)
                    actual_window_normalized = [line.strip() for line in actual_window_content]
                    search_normalized = [line.strip() for line in normalized_search_lines]

                    if actual_window_normalized == search_normalized:
                        start_idx = current_search_idx
                        found_match = True
                        logging.info(f"在行 {start_idx + 1} 找到精确匹配。")
                        break

            if not found_match:
                mismatch_info = []
                actual_content_for_error = original_lines[start_idx : min(start_idx + search_len, original_len)]
                actual_stripped_for_error = [line.strip() for line in actual_content_for_error]
                search_stripped_for_error = [line.strip() for line in normalized_search_lines]

                max_len = max(len(actual_stripped_for_error), len(search_stripped_for_error))
                for i in range(max_len):
                    actual_l = actual_stripped_for_error[i] if i < len(actual_stripped_for_error) else "<实际内容结束>"
                    search_l = search_stripped_for_error[i] if i < len(search_stripped_for_error) else "<期望内容结束>"
                    if actual_l != search_l:
                        mismatch_info.append(f"  行 {start_line_num + i}: 实际='{actual_l}', 期望='{search_l}'")
                if len(actual_stripped_for_error) != len(search_stripped_for_error):
                    mismatch_info.append(f"  行数不匹配: 实际={len(actual_stripped_for_error)}, 期望={len(search_stripped_for_error)}")

                error_msg = (
                    f"Diff 块中搜索内容与文件 '{path}' 中的实际内容不精确匹配，从第 {start_line_num} 行开始或附近。\n"
                    f"搜索内容（提供用于比较的规范化形式）:\n{''.join(normalized_search_lines)}\n"
                    f"实际内容（提供用于比较的规范化形式）:\n{''.join(actual_content_for_error)}\n"
                    f"不匹配详情:\n{chr(10).join(mismatch_info)}\n"
                    f"尝试在行号 {start_line_num} 的 +/- {search_window} 范围内搜索，但未找到精确匹配。"
                )
                logging.error(error_msg)
                raise ValueError(error_msg)

            operations.append(
                {
                    "start_idx": start_idx,
                    "search_len": search_len,
                    "replace_content": replace_content,
                }
            )
        else:
            current_line_in_diff += 1

    logging.info(f"成功解析 {len(operations)} 个 diff 块。开始应用更改。")
    operations.sort(key=lambda x: x["start_idx"], reverse=True)

    for i, op in enumerate(operations):
        start_idx = op["start_idx"]
        search_len = op["search_len"]
        replace_content = op["replace_content"]
        logging.info(f"正在应用第 {i + 1} 个 diff 块: 从行 {start_idx + 1} 开始替换 {search_len} 行。")

        replace_lines = replace_content.splitlines(keepends=True)
        if replace_content and not replace_content.endswith("\n") and replace_lines:
            # Ensure the last line added does not have an extra newline if the original replacement content didn't end with one.
            replace_lines[-1] = replace_lines[-1].rstrip("\n")

        del current_lines[start_idx : start_idx + search_len]
        current_lines[start_idx:start_idx] = replace_lines

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(current_lines)
        logging.info(f"文件 '{path}' 已成功应用 {len(operations)} 个 diff 块并保存。")
        return f"文件 '{path}' 已成功应用 {len(operations)} 个 diff 块。"
    except IOError as e:
        logging.error(f"写入文件 '{path}' 时发生 IO 错误: {e}")
        raise IOError(f"无法写入文件 '{path}': {e}")
