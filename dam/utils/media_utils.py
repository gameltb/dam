import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


class TranscodeError(Exception):
    """Custom exception for transcoding errors."""

    pass


def _run_command(command: List[str]) -> Tuple[str, str]:
    """
    Runs a shell command and returns its stdout and stderr.
    Raises TranscodeError if the command fails.
    """
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # Raises CalledProcessError on non-zero exit codes
            encoding="utf-8",  # Ensure consistent encoding
        )
        return process.stdout, process.stderr
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Command '{' '.join(e.cmd)}' failed with exit code {e.returncode}.\n"
            f"Stdout:\n{e.stdout}\n"
            f"Stderr:\n{e.stderr}"
        )
        raise TranscodeError(error_message) from e
    except FileNotFoundError as e:
        error_message = f"Command not found: {command[0]}. Please ensure it is installed and in PATH. Details: {e}"
        raise TranscodeError(error_message) from e
    except Exception as e:  # Catch any other unexpected errors
        error_message = f"An unexpected error occurred while running command '{' '.join(command)}'. Details: {e}"
        raise TranscodeError(error_message) from e


def transcode_media(
    input_path: Path,
    output_path: Path,
    tool_name: str,
    tool_params: str,
    # output_format: str # output_format is implicit in the output_path extension or tool behavior
) -> Path:
    """
    Transcodes a media file using the specified tool and parameters.

    Args:
        input_path: Path to the input media file.
        output_path: Path to save the transcoded media file.
        tool_name: The transcoding tool to use (e.g., "ffmpeg", "cjxl", "avifenc").
        tool_params: Parameters string for the transcoding tool.
                     IMPORTANT: Use {input} and {output} placeholders for file paths.

    Returns:
        Path to the transcoded output file.

    Raises:
        TranscodeError: If the transcoding fails or tool is not found.
        ValueError: If tool_params does not contain {input} and {output} placeholders.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if "{input}" not in tool_params or "{output}" not in tool_params:
        raise ValueError("tool_params must contain {input} and {output} placeholders.")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Replace placeholders with actual paths
    # Ensure paths are quoted if they might contain spaces
    formatted_params = tool_params.replace("{input}", f'"{str(input_path)}"')
    formatted_params = formatted_params.replace("{output}", f'"{str(output_path)}"')

    command_parts = [tool_name] + formatted_params.split()  # Simple split, might need shlex for complex params

    # Security Note: Directly using tool_name and parts of tool_params in a command
    # can be a security risk if these values come from untrusted user input without sanitization.
    # Here, we assume tool_name and tool_params are curated (e.g., from TranscodeProfileComponent).

    print(f"Running transcoding command: {' '.join(command_parts)}")  # For logging/debugging

    # Check if the tool exists
    if not shutil.which(tool_name):
        raise TranscodeError(f"Transcoding tool '{tool_name}' not found in PATH. Please install it.")

    try:
        _run_command(command_parts)
    except TranscodeError as e:
        # If transcoding failed, clean up potentially incomplete output file
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError as unlink_e:
                # Log this, but raise the original TranscodeError
                print(f"Warning: Could not delete incomplete output file {output_path}: {unlink_e}")
        raise e  # Re-raise the original error

    if not output_path.exists() or output_path.stat().st_size == 0:
        # Additional check in case the tool reported success but produced no/empty output
        if output_path.exists():  # if empty
            output_path.unlink(missing_ok=True)
        raise TranscodeError(f"Transcoding command ran but output file {output_path} was not created or is empty.")

    print(f"Transcoding successful. Output: {output_path}")
    return output_path


# Example usage (for testing this module directly, not part of the service yet)
if __name__ == "__main__":
    # Create dummy files for testing
    # Ensure you have ffmpeg installed for this example to run
    # You might need to create a dummy input file, e.g., a small text file named 'input.txt'
    # or a small image/video if testing with actual media tools.

    current_dir = Path(__file__).parent
    dummy_input = current_dir / "dummy_input.txt"
    dummy_output_mp4 = current_dir / "dummy_output.mp4"  # ffmpeg can create this from various inputs
    dummy_output_jxl = current_dir / "dummy_output.jxl"

    if not dummy_input.exists():
        with open(dummy_input, "w") as f:
            f.write("This is a dummy input file for transcoding tests.")
        print(f"Created dummy input: {dummy_input}")

    # Test 1: FFmpeg (example - requires ffmpeg and a suitable input)
    # This specific ffmpeg command might fail if input.txt is not a valid video/image source.
    # A more robust test would use a small, valid media file.
    # For now, let's assume a command that works with a text file for demonstration.
    # A common use of ffmpeg that can take any file is to create a video slideshow from an image,
    # or simply copy a stream if the input was valid.
    # Using -f lavfi -i color=c=blue:s=1280x720:d=1 to generate a dummy 1s video.

    # Example: Generate a 1-second blue video with ffmpeg
    # Note: The {input} placeholder is not used by this specific ffmpeg command generating a synthetic input.
    # We'll adapt the command to use a placeholder for output only.
    # For a real transcode, the command would be different.

    # Let's create a more realistic scenario with a dummy input file.
    # We'll "convert" the text file to a "video" using ffmpeg's drawtext filter on a color source.
    # This is just to make the {input} and {output} placeholders meaningful.

    ffmpeg_params_template = (
        "-y -f lavfi -i color=c=blue:s=320x240:d=1 "  # Synthetic 1s blue video
        "-vf \"drawtext=textfile='{input}':fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2\" "
        "-c:v libx264 -preset ultrafast -tune zerolatency -movflags +faststart "  # Basic H.264 output
        "{output}"
    )

    if shutil.which("ffmpeg"):
        print("\n--- Testing FFmpeg ---")
        try:
            # Replacing placeholders manually for this direct test
            params_ffmpeg = ffmpeg_params_template.replace("{input}", str(dummy_input))
            params_ffmpeg = params_ffmpeg.replace("{output}", str(dummy_output_mp4))

            # For the purpose of testing transcode_media, we pass the template
            transcode_media(
                input_path=dummy_input,
                output_path=dummy_output_mp4,
                tool_name="ffmpeg",
                tool_params=ffmpeg_params_template,  # Pass the template string
            )
            print(f"FFmpeg test successful. Output: {dummy_output_mp4}")
            if dummy_output_mp4.exists():
                dummy_output_mp4.unlink()  # Clean up
        except TranscodeError as e:
            print(f"FFmpeg test failed: {e}")
        except FileNotFoundError as e:
            print(f"FFmpeg test skipped, input file missing: {e}")

    else:
        print("ffmpeg not found, skipping FFmpeg test.")

    # Test 2: cjxl (example - requires cjxl and a suitable input image)
    # cjxl typically takes an image (e.g., PNG) and outputs JXL.
    # We'd need a dummy PNG for this. Let's skip if cjxl isn't present or no input.
    # For now, this part is illustrative.
    # A real test would involve creating a small PNG using Pillow, for example.
    # params_cjxl = "{input} {output} -q 90" # Example params
    if shutil.which("cjxl"):
        print("\n--- Testing cjxl (placeholder) ---")
        print("cjxl found, but test requires a dummy image input (e.g., PNG). Skipping detailed test.")
        # try:
        #     # Create a dummy PNG if Pillow is available, or assume one exists
        #     dummy_png_input = current_dir / "dummy_input.png"
        #     if not dummy_png_input.exists():
        #         print(f"Skipping cjxl test: {dummy_png_input} not found.")
        #     else:
        #         transcode_media(dummy_png_input, dummy_output_jxl, "cjxl", params_cjxl)
        #         print(f"cjxl test successful. Output: {dummy_output_jxl}")
        #         if dummy_output_jxl.exists(): dummy_output_jxl.unlink()
        # except TranscodeError as e:
        #     print(f"cjxl test failed: {e}")
        # except FileNotFoundError as e:
        #     print(f"cjxl test skipped, input file missing: {e}")
    else:
        print("cjxl not found, skipping cjxl test.")

    # Clean up dummy input
    if dummy_input.exists():
        dummy_input.unlink()
        print(f"Cleaned up dummy input: {dummy_input}")

    print("\nMedia utils tests finished.")


import logging
import os
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from dam.models.core import FileLocationComponent  # Assuming this component stores relative path
from dam.services import ecs_service

logger = logging.getLogger(__name__)


async def get_file_path_for_entity(
    session: AsyncSession,
    entity_id: int,
    base_asset_storage_path: str,  # Get this from world_config.ASSET_STORAGE_PATH
    variant_name: Optional[str] = "original",  # Or some other way to specify which file if multiple exist
) -> Optional[str]:
    """
    Retrieves the full file path for a given entity.
    This is a mock implementation and needs to be adapted to the actual
    FileLocationComponent structure and storage layout.

    Args:
        session: The SQLAlchemy async session.
        entity_id: The ID of the entity.
        base_asset_storage_path: The root path where assets are stored for the current world.
        variant_name: Specifies which variant of the file to retrieve (e.g., "original", "thumbnail").
                      The logic to handle variants needs to be implemented based on how they are stored.

    Returns:
        The full path to the file, or None if not found.
    """
    logger.debug(f"Attempting to get file path for entity {entity_id}, variant {variant_name}")

    file_location_comps = await ecs_service.get_components(session, entity_id, FileLocationComponent)

    if not file_location_comps:
        logger.warning(f"No FileLocationComponent found for entity {entity_id}.")
        return None

    target_file_loc: Optional[FileLocationComponent] = None

    if len(file_location_comps) == 1:
        target_file_loc = file_location_comps[0]
    else:
        # This logic needs to be adapted based on how FileLocationComponent stores variant info.
        # For example, if it has a 'variant_type' or 'name' field:
        # target_file_loc = next((flc for flc in file_location_comps if flc.variant_name == variant_name), None)
        # For now, defaulting to the first one if specific variant logic isn't implemented/matched.
        logger.debug(
            f"Multiple FileLocationComponents found for entity {entity_id}. Attempting to find variant '{variant_name}'. Defaulting to first if not specific match."
        )
        # Pseudo-code for variant matching:
        # if hasattr(FileLocationComponent, 'variant_name_column'): # Replace with actual attribute name
        #     target_file_loc = next((flc for flc in file_location_comps if getattr(flc, 'variant_name_column', None) == variant_name), None)

        if not target_file_loc:  # If no specific match or variant logic not fully implemented here
            target_file_loc = file_location_comps[0]

    if not target_file_loc:
        logger.warning(f"No suitable FileLocationComponent found for entity {entity_id} and variant '{variant_name}'.")
        return None

    relative_path = getattr(target_file_loc, "stored_path_relative", None)

    if not relative_path:
        logger.error(
            f"FileLocationComponent for entity {entity_id} (variant {variant_name}, component_id {target_file_loc.id}) does not have a 'stored_path_relative' attribute or it's empty."
        )
        return None

    full_path = os.path.join(base_asset_storage_path, relative_path)

    # It's good practice to check if the file actually exists
    if not os.path.exists(full_path):
        logger.warning(
            f"File path constructed for entity {entity_id} (variant {variant_name}) does not exist: {full_path} (from relative: {relative_path})"
        )
        # Return None if physical file doesn't exist, as the path is invalid for processing
        return None

    logger.info(f"Resolved file path for entity {entity_id} (variant {variant_name}): {full_path}")
    return full_path


__all__ = ["transcode_media", "TranscodeError", "get_file_path_for_entity"]
