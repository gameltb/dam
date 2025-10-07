"""A tool for applying diffs to files."""
import logging
import pathlib
import re
from typing import TypedDict

from domarkx.tools.tool_factory import tool_handler

logger = logging.getLogger(__name__)


class Operation(TypedDict):
    """Represents a single search-and-replace operation."""

    start_idx: int
    search_len: int
    replace_content: str


def _read_and_validate_file(path: str) -> tuple[pathlib.Path, list[str]]:
    """Read and validate the file path."""
    logger.info("Attempting to apply diff to file '%s'.", path)
    file_path = pathlib.Path(path)
    if not file_path.exists():
        logger.error("File '%s' does not exist.", path)
        raise FileNotFoundError(f"File '{path}' does not exist.")
    if not file_path.is_file():
        logger.error("Path '%s' is a directory, not a file.", path)
        raise IsADirectoryError(f"Path '{path}' is a directory, not a file.")

    try:
        with file_path.open(encoding="utf-8") as f:
            original_lines = f.readlines()
        logger.info("Successfully read file '%s'.", path)
        return file_path, original_lines
    except OSError as e:
        logger.error("IO error occurred while reading file '%s': %s", path, e)
        raise OSError(f"Could not read file '{path}': {e}") from e


def _parse_diff_blocks(diff_lines: list[str], original_lines: list[str], path: str) -> list[Operation]:
    """Parse diff blocks from the diff string."""
    operations: list[Operation] = []
    current_line_in_diff = 0
    logger.info("Starting to parse diff blocks.")
    while current_line_in_diff < len(diff_lines):
        line = diff_lines[current_line_in_diff]

        if not re.fullmatch(r"\s*<<<<<<< ?SEARCH\s*", line.strip()):
            current_line_in_diff += 1
            continue

        logger.info("Found SEARCH block, starting at diff line %d.", current_line_in_diff + 1)
        current_line_in_diff += 1
        if current_line_in_diff >= len(diff_lines):
            raise ValueError("Invalid diff format: SEARCH block missing content after delimiter.")

        line = diff_lines[current_line_in_diff].strip()
        if not line.startswith(":start_line:"):
            raise ValueError("Invalid diff format: SEARCH block missing ':start_line:' after delimiter.")
        try:
            start_line_num = int(line.split(":")[2].strip())
        except (IndexError, ValueError) as e:
            raise ValueError("Invalid diff format: Invalid line number after ':start_line:'.") from e

        current_line_in_diff += 1

        if current_line_in_diff >= len(diff_lines) or not re.fullmatch(
            r"\s*-------\s*", diff_lines[current_line_in_diff].strip()
        ):
            raise ValueError("Invalid diff format: SEARCH block missing '-------' separator after start line.")
        current_line_in_diff += 1

        search_content_lines: list[str] = []
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

        replace_content_lines: list[str] = []
        while current_line_in_diff < len(diff_lines):
            line = diff_lines[current_line_in_diff]
            if re.fullmatch(r"\s*>{5,9} ?REPLACE\s*", line.strip()):
                break
            replace_content_lines.append(line)
            current_line_in_diff += 1
        replace_content = "".join(replace_content_lines)

        if current_line_in_diff >= len(diff_lines) or not re.fullmatch(
            r"\s*>{5,9} ?REPLACE\s*", diff_lines[current_line_in_diff].strip()
        ):
            raise ValueError("Invalid diff format: REPLACE content missing '>>>>>>> REPLACE' end marker.")
        current_line_in_diff += 1

        start_idx = start_line_num - 1
        normalized_search_lines: list[str] = search_content.splitlines(keepends=True)

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
            logger.info(
                "Searching for matching content in file '%s', around line %d +/- %d.",
                path,
                start_line_num,
                search_window,
            )
            for current_search_idx in range(
                max(0, start_idx - search_window), min(original_len - search_len + 1, start_idx + search_window + 1)
            ):
                actual_window_content = original_lines[current_search_idx : current_search_idx + search_len]
                actual_window_normalized = [line.strip() for line in actual_window_content]
                search_normalized = [line.strip() for line in normalized_search_lines]
                if actual_window_normalized == search_normalized:
                    start_idx = current_search_idx
                    found_match = True
                    logger.info("Exact match found at line %d.", start_idx + 1)
                    break

        if not found_match:
            mismatch_info: list[str] = []
            actual_content_for_error = original_lines[start_idx : min(start_idx + search_len, original_len)]
            actual_stripped_for_error: list[str] = [line.strip() for line in actual_content_for_error]
            search_stripped_for_error: list[str] = [line.strip() for line in normalized_search_lines]

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
            logger.error(error_msg)
            raise ValueError(error_msg)

        operations.append(
            {
                "start_idx": start_idx,
                "search_len": search_len,
                "replace_content": replace_content,
            }
        )

    logger.info("Successfully parsed %d diff blocks.", len(operations))
    return operations


def _apply_operations(current_lines: list[str], operations: list[Operation]) -> list[str]:
    """Apply the operations to the lines of the file."""
    operations.sort(key=lambda x: x["start_idx"], reverse=True)
    for i, op in enumerate(operations):
        start_idx = op["start_idx"]
        search_len = op["search_len"]
        replace_content = op["replace_content"]
        logger.info(
            "Applying diff block %d: Replacing %d lines starting at line %d.", i + 1, search_len, start_idx + 1
        )

        replace_lines = replace_content.splitlines(keepends=True)
        if replace_content and not replace_content.endswith("\n") and replace_lines:
            replace_lines[-1] = replace_lines[-1].rstrip("\n")

        del current_lines[start_idx : start_idx + search_len]
        current_lines[start_idx:start_idx] = replace_lines
    return current_lines


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
    file_path, original_lines = _read_and_validate_file(path)
    diff_lines = diff.splitlines(keepends=True)
    operations = _parse_diff_blocks(diff_lines, original_lines, path)
    current_lines = _apply_operations(list(original_lines), operations)

    try:
        with file_path.open("w", encoding="utf-8") as f:
            f.writelines(current_lines)
        logger.info("File '%s' successfully updated with %d diff blocks applied.", path, len(operations))
        return f"File '{path}' successfully updated with {len(operations)} diff blocks applied."
    except OSError as e:
        logger.error("IO error occurred while writing to file '%s': %s", path, e)
        raise OSError(f"Could not write to file '{path}': {e}") from e
