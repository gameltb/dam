"""A tool for reading files, including partial content and multiple files via wildcards."""

import glob
import logging
import pathlib

from domarkx.tools.tool_factory import tool_handler

logger = logging.getLogger(__name__)


def _read_single_file(file_path_str: str, s_line: int | None, e_line: int | None) -> str:
    """Read a single file, with an optional line range."""
    logger.info("Reading single file: '%s'", file_path_str)
    file_path = pathlib.Path(file_path_str)

    if not file_path.exists():
        msg = f"File '{file_path}' does not exist."
        logger.error(msg)
        raise FileNotFoundError(msg)
    if not file_path.is_file():
        msg = f"Path '{file_path}' is a directory, not a file."
        logger.error(msg)
        raise IsADirectoryError(msg)

    # Simulate PDF/DOCX handling
    if file_path.suffix.lower() in [".pdf", ".docx"]:
        logger.warning("Note: '%s' is a binary file (PDF/DOCX). This tool simulates raw text extraction.", file_path)
        return (
            f"Note: '{file_path}' is a binary file (PDF/DOCX). "
            f"This tool simulates raw text extraction. Actual implementation would use appropriate libraries.\n"
            f"Simulated content: This is the extracted text from {file_path}."
        )

    try:
        # read_text is simpler and safer for reading entire files
        lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
        logger.info("Successfully read %d lines from file '%s'.", len(lines), file_path)

        # If no line numbers are specified, return the whole content
        if s_line is None and e_line is None:
            return "".join(lines)

        start_idx = (s_line - 1) if s_line is not None and s_line > 0 else 0
        end_idx = e_line if e_line is not None and e_line > 0 else len(lines)
        start_idx = max(0, min(start_idx, len(lines)))
        end_idx = max(0, min(end_idx, len(lines)))

        if start_idx >= end_idx:
            msg = (
                f"The specified line range (from {s_line} to {e_line}) is invalid or empty, "
                f"as the file only has {len(lines)} lines. No content was read."
            )
            logger.warning(msg)
            raise ValueError(msg)

        output_lines = lines[start_idx:end_idx]
        logger.info("Successfully extracted specified lines from file '%s'.", file_path)
        return "".join(output_lines)

    except PermissionError as e:
        msg = f"No permission to read file '{file_path}': {e}"
        logger.error(msg)
        raise PermissionError(msg) from e
    except OSError as e:
        msg = f"IO error occurred while reading file '{file_path}': {e}"
        logger.error(msg)
        raise OSError(msg) from e
    except Exception as e:
        msg = f"Unexpected error occurred while reading file '{file_path}': {e}"
        logger.error(msg)
        raise Exception(msg) from e


def _read_multiple_files(file_list: list[str]) -> str:
    """Read multiple files and concatenate their contents."""
    results: list[str] = []
    for file_item in sorted(file_list):
        try:
            # When reading multiple files, we read the whole file, so line numbers are None
            file_content = _read_single_file(file_item, None, None)
            results.append(f"--- File: {file_item} ---\n{file_content}\n")
        except (OSError, FileNotFoundError, IsADirectoryError, PermissionError, ValueError, Exception) as e:
            logger.error("Error processing file '%s': %s", file_item, e)
            results.append(f"--- File: {file_item} ---\nError: {e}\n")
    if not results:
        msg = "All specified files could not be read or do not exist."
        logger.error(msg)
        raise RuntimeError(msg)
    return "\n".join(results)


@tool_handler()
def tool_read_file(path: str | list[str], start_line: int | None = None, end_line: int | None = None) -> str:
    """
    Read file content, with support for partial reads and wildcards.

    When using wildcards or a list of files, `start_line` and `end_line` are ignored.

    Args:
        path: File path(s) to read. Can be a single path, a list, or a string with wildcards.
        start_line: Optional start line number (1-based).
        end_line: Optional end line number (1-based, inclusive).

    Returns:
        The file content, or concatenated contents for multiple files.

    Raises:
        TypeError: If argument types are incorrect.
        FileNotFoundError: If a file does not exist.
        IsADirectoryError: If a path is a directory.
        PermissionError: If lacking permission to read a file.
        ValueError: If the line range is invalid.
        RuntimeError: If no files could be read.

    """
    start_line = int(start_line) if start_line else None
    end_line = int(end_line) if end_line else None
    logger.info("Attempting to read file: '%s'. Start line: %s, End line: %s.", path, start_line, end_line)

    if isinstance(path, list):
        logger.info("Path is a list, processing %d files.", len(path))
        return _read_multiple_files(path)

    if not isinstance(path, str):
        msg = f"Argument 'path' must be a string or list of strings, but received {type(path).__name__}."
        logger.error(msg)
        raise TypeError(msg)

    # It's a string, check for wildcards
    if not glob.has_magic(path):
        return _read_single_file(path, start_line, end_line)

    logger.info("Path '%s' contains wildcards, will process multiple files.", path)
    # Using Path.glob/rglob for wildcard matching
    base_dir = pathlib.Path.cwd()
    matching_files = [str(p) for p in base_dir.rglob(path) if p.is_file()]

    logger.info("Found %d files matching '%s'.", len(matching_files), path)
    if not matching_files:
        warning_msg = f"Warning: No files found matching wildcard pattern '{path}'."
        logger.warning(warning_msg)
        return warning_msg

    return _read_multiple_files(matching_files)
