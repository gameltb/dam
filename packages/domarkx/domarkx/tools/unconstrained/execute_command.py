import logging
import os
import subprocess


def execute_command(command: str, cwd: str = None) -> str:
    """
    Execute a CLI command on the system.

    Args:
        command (str): CLI command to execute. For commands with special characters or newlines, use `$'...'` quoting to avoid shell parsing errors.
        cwd (str): Working directory for the command (defaults to current directory).

    Returns:
        str: stdout and stderr output of the command.

    Raises:
        TypeError: If 'command' or 'cwd' argument types are incorrect.
        FileNotFoundError: If the specified working directory does not exist.
        NotADirectoryError: If the specified working directory is not a directory.
        RuntimeError: If the command fails (non-zero exit code).
        Exception: For other unexpected errors.
    """
    logging.info(f"Attempting to execute command: '{command}' in directory '{cwd if cwd else '.'}'.")

    if not isinstance(command, str):
        error_msg = f"Argument 'command' must be a string, but received {type(command).__name__}."
        logging.error(error_msg)
        raise TypeError(error_msg)

    if cwd is not None and not isinstance(cwd, str):
        error_msg = f"Argument 'cwd' must be a string or None, but received {type(cwd).__name__}."
        logging.error(error_msg)
        raise TypeError(error_msg)

    effective_cwd = cwd if cwd else "."

    if not os.path.exists(effective_cwd):
        logging.error(f"Specified working directory '{effective_cwd}' does not exist.")
        raise FileNotFoundError(f"Specified working directory '{effective_cwd}' does not exist.")
    if not os.path.isdir(effective_cwd):
        logging.error(f"Specified working directory '{effective_cwd}' is not a directory.")
        raise NotADirectoryError(f"Specified working directory '{effective_cwd}' is not a directory.")
    logging.info(f"Command working directory will be: '{effective_cwd}'.")

    try:
        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=effective_cwd,
            check=False,
        )

        output_parts = []
        if process.stdout.strip():
            output_parts.append(f"--- Command Output (stdout) ---\n{process.stdout.strip()}\n")
        if process.stderr.strip():
            output_parts.append(f"--- Command Error (stderr) ---\n{process.stderr.strip()}\n")
        result = "".join(output_parts)
        if process.returncode != 0:
            error_msg = f"Command failed with exit code {process.returncode}.\n{result}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        logging.info(f"Command executed successfully.\n{result}")
        return result
    except Exception as e:
        error_msg = f"Unexpected error occurred while executing command: {e}"
        logging.error(error_msg)
        raise Exception(error_msg)
