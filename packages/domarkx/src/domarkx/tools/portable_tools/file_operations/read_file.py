import glob
import logging
import pathlib


def tool_read_file(
    path: str | list[str], start_line: int | None = None, end_line: int | None = None
) -> str:
    """
    Read the content of the specified file path. Supports reading partial content by specifying `start_line` and `end_line`.
    Supports wildcard `*` to read multiple files at once (e.g., `*.py`, `dir/*.md`).
    When using wildcards or passing a file list, `start_line` and `end_line` are ignored and all matching files are read in full.

    Args:
        path (Union[str, List[str]]): File path to read. Can be a single path string (with wildcards),
                                     or a list of file paths. Use `**` for recursive directory matching.
        start_line (int): (Optional) Start line number (1-based).
        end_line (int): (Optional) End line number (1-based, inclusive).

    Returns:
        str: File content. If multiple files are matched or a file list is provided, each file's content is shown, separated by filename.

    Raises:
        TypeError: If argument types are incorrect.
        FileNotFoundError: If file does not exist.
        IsADirectoryError: If path is a directory.
        PermissionError: If lacking permission to read file.
        ValueError: If line range is invalid.
        IOError: If file read/write error occurs.
        Exception: For other unexpected errors.

    """
    start_line = int(start_line) if start_line else None
    end_line = int(end_line) if end_line else None
    logging.info(f"Attempting to read file: '{path}'. Start line: {start_line}, End line: {end_line}.")

    def _read_single_file(file_path_str: str, s_line: int | None, e_line: int | None) -> str:
        logging.info(f"Reading single file: '{file_path_str}'.")
        file_path = pathlib.Path(file_path_str)

        if not file_path.exists():
            logging.error(f"File '{file_path}' does not exist.")
            raise FileNotFoundError(f"File '{file_path}' does not exist.")
        if not file_path.is_file():
            logging.error(f"Path '{file_path}' is a directory, not a file.")
            raise IsADirectoryError(f"Path '{file_path}' is a directory, not a file.")

        # Simulate PDF/DOCX handling
        if file_path.suffix.lower() in [".pdf", ".docx"]:
            logging.warning(
                f"Note: '{file_path}' is a binary file (PDF/DOCX). This tool simulates raw text extraction."
            )
            return f"Note: '{file_path}' is a binary file (PDF/DOCX). This tool simulates raw text extraction. Actual implementation would use appropriate libraries.\nSimulated content: This is the extracted text from {file_path}."

        try:
            with file_path.open(encoding="utf-8") as f:
                lines = f.readlines()
            logging.info(f"Successfully read {len(lines)} lines from file '{file_path}'.")

            output_lines: list[str] = []
            start_idx = (s_line - 1) if s_line is not None and s_line > 0 else 0
            end_idx = e_line if e_line is not None and e_line > 0 else len(lines)
            start_idx = max(0, min(start_idx, len(lines)))
            end_idx = max(0, min(end_idx, len(lines)))

            if start_idx >= end_idx and s_line is not None and e_line is not None:
                error_msg = f"The specified line range (from {s_line} to {e_line}) is invalid or empty, as the file only has {len(lines)} lines. No content was read."
                logging.warning(error_msg)
                raise ValueError(error_msg)

            for i in range(start_idx, end_idx):
                output_lines.append(lines[i].rstrip("\n"))

            logging.info(f"Successfully extracted specified lines from file '{file_path}'.")
            return "\n".join(output_lines)
        except PermissionError as e:
            error_msg = f"No permission to read file '{file_path}': {e}"
            logging.error(error_msg)
            raise PermissionError(error_msg) from e
        except OSError as e:
            error_msg = f"IO error occurred while reading file '{file_path}': {e}"
            logging.error(error_msg)
            raise OSError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error occurred while reading file '{file_path}': {e}"
            logging.error(error_msg)
            raise Exception(error_msg) from e

    def _read_multiple_files(file_list: list[str]) -> str:
        results: list[str] = []
        for file_item in sorted(file_list):
            try:
                file_content = _read_single_file(file_item, None, None)
                results.append(f"--- File: {file_item} ---\n{file_content}\n")
            except (OSError, FileNotFoundError, IsADirectoryError, PermissionError, ValueError, Exception) as e:
                logging.error(f"Error processing file '{file_item}': {e}")
                results.append(f"--- File: {file_item} ---\nError: {e}\n")
        if not results:
            error_msg = "All specified files could not be read or do not exist."
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        return "\n".join(results)

    if isinstance(path, list):
        logging.info(f"Path argument is a list, will process {len(path)} files.")
        return _read_multiple_files(path)
    if isinstance(path, str):
        if glob.has_magic(path):
            logging.info(f"Path '{path}' contains wildcards, will process multiple files.")
            use_recursive = "**" in path
            p = pathlib.Path(path)
            full_glob_pattern = str(p)
            matching_files = glob.glob(full_glob_pattern, recursive=use_recursive)
            logging.info(f"Found {len(matching_files)} files matching '{path}' (recursive: {use_recursive}).")
            if not matching_files:
                warning_msg = f"Warning: No files found matching wildcard pattern '{path}'."
                logging.warning(warning_msg)
                return warning_msg
            return _read_multiple_files(matching_files)
        return _read_single_file(path, start_line, end_line)
    error_msg = f"Argument 'path' must be a string or list of strings, but received {type(path).__name__}."
    logging.error(error_msg)
    raise TypeError(error_msg)
