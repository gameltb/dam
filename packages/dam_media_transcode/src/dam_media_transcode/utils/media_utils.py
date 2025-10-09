"""Provides utility functions for transcoding media files using external tools."""

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscodeError(Exception):
    """Custom exception for transcoding errors."""


def _run_command(command: list[str]) -> tuple[str, str]:
    """
    Run a shell command and return its stdout and stderr.

    Raises:
        TranscodeError: If the command fails.

    """
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
        return process.stdout, process.stderr
    except subprocess.CalledProcessError as e:
        error_message = f"Command '{' '.join(e.cmd)}' failed with exit code {e.returncode}.\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}"
        raise TranscodeError(error_message) from e
    except FileNotFoundError as e:
        error_message = f"Command not found: {command[0]}. Please ensure it is installed and in PATH. Details: {e}"
        raise TranscodeError(error_message) from e
    except Exception as e:
        error_message = f"An unexpected error occurred while running command '{' '.join(command)}'. Details: {e}"
        raise TranscodeError(error_message) from e


def transcode_media(
    input_path: Path,
    output_path: Path,
    tool_name: str,
    tool_params: str,
) -> Path:
    """
    Transcode a media file using the specified tool and parameters.

    Args:
        input_path: Path to the input media file.
        output_path: Path to save the transcoded media file.
        tool_name: The transcoding tool to use (e.g., "ffmpeg").
        tool_params: Parameters string for the tool. Use {input} and {output}.

    Returns:
        Path to the transcoded output file.

    Raises:
        TranscodeError: If the transcoding fails or tool is not found.
        ValueError: If tool_params is missing placeholders.

    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if "{input}" not in tool_params or "{output}" not in tool_params:
        raise ValueError("tool_params must contain {input} and {output} placeholders.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    formatted_params = tool_params.replace("{input}", f'"{input_path!s}"')
    formatted_params = formatted_params.replace("{output}", f'"{output_path!s}"')

    command_parts = [tool_name, *formatted_params.split()]

    logger.info("Running transcoding command: %s", " ".join(command_parts))

    if not shutil.which(tool_name):
        raise TranscodeError(f"Transcoding tool '{tool_name}' not found in PATH. Please install it.")

    try:
        _run_command(command_parts)
    except TranscodeError as e:
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError as unlink_e:
                logger.warning("Could not delete incomplete output file %s: %s", output_path, unlink_e)
        raise e

    if not output_path.exists() or output_path.stat().st_size == 0:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise TranscodeError(f"Transcoding command ran but output file {output_path} was not created or is empty.")

    logger.info("Transcoding successful. Output: %s", output_path)
    return output_path


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    dummy_input = current_dir / "dummy_input.txt"
    dummy_output_mp4 = current_dir / "dummy_output.mp4"
    dummy_output_jxl = current_dir / "dummy_output.jxl"

    if not dummy_input.exists():
        with dummy_input.open("w") as f:
            f.write("This is a dummy input file for transcoding tests.")
        logger.info("Created dummy input: %s", dummy_input)

    ffmpeg_params_template = (
        "-y -f lavfi -i color=c=blue:s=320x240:d=1 "
        "-vf \"drawtext=textfile='{input}':fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2\" "
        "-c:v libx264 -preset ultrafast -tune zerolatency -movflags +faststart "
        "{output}"
    )

    if shutil.which("ffmpeg"):
        logger.info("\n--- Testing FFmpeg ---")
        try:
            transcode_media(
                input_path=dummy_input,
                output_path=dummy_output_mp4,
                tool_name="ffmpeg",
                tool_params=ffmpeg_params_template,
            )
            logger.info("FFmpeg test successful. Output: %s", dummy_output_mp4)
            if dummy_output_mp4.exists():
                dummy_output_mp4.unlink()
        except TranscodeError as e:
            logger.error("FFmpeg test failed: %s", e)
        except FileNotFoundError as e:
            logger.error("FFmpeg test skipped, input file missing: %s", e)
    else:
        logger.info("ffmpeg not found, skipping FFmpeg test.")

    if shutil.which("cjxl"):
        logger.info("\n--- Testing cjxl (placeholder) ---")
        logger.info("cjxl found, but test requires a dummy image input (e.g., PNG). Skipping detailed test.")
    else:
        logger.info("cjxl not found, skipping cjxl test.")

    if dummy_input.exists():
        dummy_input.unlink()
        logger.info("Cleaned up dummy input: %s", dummy_input)

    logger.info("\nMedia utils tests finished.")


__all__ = ["TranscodeError", "transcode_media"]
