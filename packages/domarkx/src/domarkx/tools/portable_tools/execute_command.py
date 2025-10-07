import logging
import subprocess


def tool_execute_command(command: str) -> str:
    """
    Executes a shell command and returns its output.

    Args:
        command (str): The command to execute.

    Returns:
        str: The output of the command.

    Raises:
        subprocess.CalledProcessError: If the command fails.

    """
    logging.info(f"Executing command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing command: {e}\n{e.stderr}")
        raise e
