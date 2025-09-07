import logging
import os
import re
from typing import TypedDict

from domarkx.tools.tool_factory import tool_handler


class Operation(TypedDict):
    start_idx: int
    search_len: int
    replace_content: str


@tool_handler()
def apply_diff_tool(path: str, diff: str) -> str:
    """
    Apply search and replace blocks to modify a file.

    Args:
        path (str): Path to the file to modify.
        diff (str): Search/replace block defining the changes.

    Returns:
        str: Result message after applying the diff.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path is a directory.
        IOError: If file read/write error occurs.
        ValueError: If diff format is invalid or does not match file content.
        Exception: For other unexpected errors.
    """
    logging.info(f"Attempting to apply diff to file '{path}'.")

    if not os.path.exists(path):
        logging.error(f"File '{path}' does not exist.")
        raise FileNotFoundError(f"File '{path}' does not exist.")
    if not os.path.isfile(path):
        logging.error(f"Path '{path}' is a directory, not a file.")
        raise IsADirectoryError(f"Path '{path}' is a directory, not a file.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            original_lines = f.readlines()
        logging.info(f"Successfully read file '{path}'.")
    except IOError as e:
        logging.error(f"IO error occurred while reading file '{path}': {e}")
        raise IOError(f"Could not read file '{path}': {e}")

    current_lines = list(original_lines)  # Create a copy for modification
    operations: list[Operation] = []
    diff_lines = diff.splitlines(keepends=True)
    current_line_in_diff = 0

    logging.info("Starting to parse diff blocks.")
    while current_line_in_diff < len(diff_lines):
        line = diff_lines[current_line_in_diff]

        if re.fullmatch(r"\s*<<<<<<< ?SEARCH\s*", line.strip()):
            logging.info(f"Found SEARCH block, starting at diff line {current_line_in_diff + 1}.")
            current_line_in_diff += 1
            if current_line_in_diff >= len(diff_lines):
                raise ValueError("Invalid diff format: SEARCH block missing content after delimiter.")

            line = diff_lines[current_line_in_diff].strip()
            if not line.startswith(":start_line:"):
                raise ValueError("Invalid diff format: SEARCH block missing ':start_line:' after delimiter.")
            try:
                start_line_num = int(line.split(":")[2].strip())
            except (IndexError, ValueError):
                raise ValueError("Invalid diff format: Invalid line number after ':start_line:'.")

            current_line_in_diff += 1

            if current_line_in_diff >= len(diff_lines) or not re.fullmatch(
                r"\s*-------\s*", diff_lines[current_line_in_diff].strip()
            ):
                raise ValueError("Invalid diff format: SEARCH block missing '-------' separator after start line.")
            current_line_in_diff += 1

            search_content_lines = []
            while current_line_in_diff < len(diff_lines) and not re.fullmatch(
                r"\s*=======\s*", diff_lines[current_line_in_diff].strip()
            ):
                search_content_lines.append(diff_lines[current_line_in_diff])
                current_line_in_diff += 1
            search_content = "".join(search_content_lines)

            if current_line_in_diff >= len(diff_lines) or not re.fullmatch(
                r"\s*=======\s*", diff_lines[current_line_in_diff].strip()
            ):
                raise ValueError("Invalid diff format: SEARCH content missing '=======' separator.")
            current_line_in_diff += 1

            replace_content_lines = []
            while current_line_in_diff < len(diff_lines):
                line = diff_lines[current_line_in_diff]
                if re.fullmatch(r"\s*>{5,9} ?REPLACE\s*", line.strip()):
                    break  # Found end marker
                replace_content_lines.append(line)
                current_line_in_diff += 1
            replace_content = "".join(replace_content_lines)

            if current_line_in_diff >= len(diff_lines) or not re.fullmatch(
                r"\s*>{5,9} ?REPLACE\s*", diff_lines[current_line_in_diff].strip()
            ):
                raise ValueError("Invalid diff format: REPLACE content missing '>>>>>>> REPLACE' end marker.")
            current_line_in_diff += 1

            start_idx = start_line_num - 1
            # Use splitlines(keepends=True) to preserve original line endings for accurate comparison
            normalized_search_lines = search_content.splitlines(keepends=True)

            if start_idx < 0 or start_idx > len(original_lines):
                raise ValueError(
                    f"Start line number {start_line_num} in diff block is out of bounds for file '{path}' (1 to {len(original_lines) + 1})."
                )

            search_window = 5
            found_match = False
            original_len = len(original_lines)
            search_len = len(normalized_search_lines)

            if search_len == 0 and search_content.strip() != "":
                raise ValueError("Search content could not be parsed into a valid list of lines.")
            if search_len == 0 and search_content.strip() == "":
                found_match = True
            else:
                logging.info(
                    f"Searching for matching content in file '{path}', around line {start_line_num} +/- {search_window}."
                )
                for current_search_idx in range(
                    max(0, start_idx - search_window), min(original_len - search_len + 1, start_idx + search_window + 1)
                ):
                    actual_window_content = original_lines[current_search_idx : current_search_idx + search_len]

                    # Normalize both for comparison (strip leading/trailing whitespace)
                    actual_window_normalized = [line.strip() for line in actual_window_content]
                    search_normalized = [line.strip() for line in normalized_search_lines]

                    if actual_window_normalized == search_normalized:
                        start_idx = current_search_idx
                        found_match = True
                        logging.info(f"Exact match found at line {start_idx + 1}.")
                        break

            if not found_match:
                mismatch_info = []
                actual_content_for_error = original_lines[start_idx : min(start_idx + search_len, original_len)]
                actual_stripped_for_error = [line.strip() for line in actual_content_for_error]
                search_stripped_for_error = [line.strip() for line in normalized_search_lines]

                max_len = max(len(actual_stripped_for_error), len(search_stripped_for_error))
                for i in range(max_len):
                    actual_l = (
                        actual_stripped_for_error[i] if i < len(actual_stripped_for_error) else "<Actual content ends>"
                    )
                    search_l = (
                        search_stripped_for_error[i]
                        if i < len(search_stripped_for_error)
                        else "<Expected content ends>"
                    )
                    if actual_l != search_l:
                        mismatch_info.append(f"  Line {start_line_num + i}: Actual='{actual_l}', Expected='{search_l}'")
                if len(actual_stripped_for_error) != len(search_stripped_for_error):
                    mismatch_info.append(
                        f"  Line count mismatch: Actual={len(actual_stripped_for_error)}, Expected={len(search_stripped_for_error)}"
                    )

                error_msg = (
                    f"Search content in diff block does not match the actual content in file '{path}', starting or around line {start_line_num}.\n"
                    f"Search content (normalized for comparison):\n{''.join(normalized_search_lines)}\n"
                    f"Actual content (normalized for comparison):\n{''.join(actual_content_for_error)}\n"
                    f"Mismatch details:\n{chr(10).join(mismatch_info)}\n"
                    f"Tried to find an exact match within +/- {search_window} lines of line number {start_line_num}, but no match was found."
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

    logging.info(f"Successfully parsed {len(operations)} diff blocks. Starting to apply changes.")
    operations.sort(key=lambda x: x["start_idx"], reverse=True)

    for i, op in enumerate(operations):
        start_idx = op["start_idx"]
        search_len = op["search_len"]
        replace_content = op["replace_content"]
        logging.info(f"Applying diff block {i + 1}: Replacing {search_len} lines starting at line {start_idx + 1}.")

        replace_lines = replace_content.splitlines(keepends=True)
        if replace_content and not replace_content.endswith("\n") and replace_lines:
            # Ensure the last line added does not have an extra newline if the original replacement content didn't end with one.
            replace_lines[-1] = replace_lines[-1].rstrip("\n")

        del current_lines[start_idx : start_idx + search_len]
        current_lines[start_idx:start_idx] = replace_lines

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(current_lines)
        logging.info(f"File '{path}' successfully updated with {len(operations)} diff blocks applied.")
        return f"File '{path}' successfully updated with {len(operations)} diff blocks applied."
    except IOError as e:
        logging.error(f"IO error occurred while writing to file '{path}': {e}")
        raise IOError(f"Could not write to file '{path}': {e}")
