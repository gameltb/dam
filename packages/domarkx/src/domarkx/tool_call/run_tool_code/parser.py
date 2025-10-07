"""A robust parser for extracting tool calls from a message string."""
import re
from typing import Any


class ToolCallParsingError(Exception):
    """Custom exception for tool call parsing errors."""

    pass


def _parse_params(tool_block_content: str, tool_name: str) -> dict[str, Any]:
    """
    Parse parameters from the content of a tool block.

    Args:
        tool_block_content (str): The content of the tool block.
        tool_name (str): The name of the tool, for error reporting.

    Returns:
        dict: A dictionary of parsed parameters.

    Raises:
        ToolCallParsingError: If malformed content is found.

    """
    current_params = {}
    temp_idx = 0
    while temp_idx < len(tool_block_content):
        # Find the start tag of the next parameter
        param_open_tag_match = re.search(
            r"<\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*>", tool_block_content[temp_idx:], re.DOTALL
        )

        if not param_open_tag_match:
            break  # No more tags, or only whitespace remains, exit the loop

        # Before the current parameter tag, check for non-whitespace content.
        # This handles cases like `<tool> malformed text <param>...</param> </tool>`
        text_before_tag = tool_block_content[temp_idx : temp_idx + param_open_tag_match.start()].strip()
        if text_before_tag:
            raise ToolCallParsingError(
                f"Malformed non-tag content '{text_before_tag}' found before a tag in tool '{tool_name}' block."
            )

        param_name = param_open_tag_match.group(1)
        # Get the position after the parameter's start tag
        param_content_start = temp_idx + param_open_tag_match.end()

        # Try to find the matching end tag for the current parameter
        param_close_tag_pattern = re.compile(r"<\/\s*" + re.escape(param_name) + r"\s*>", re.DOTALL)
        param_close_tag_match = param_close_tag_pattern.search(tool_block_content, param_content_start)

        param_value_raw = ""
        # By default, if the parameter block is incomplete, its content extends to the end of the tool block
        param_block_end_pos = len(tool_block_content)

        if param_close_tag_match:
            # If a matching end tag is found, the content is between the start and end tags
            param_value_raw = tool_block_content[param_content_start : param_close_tag_match.start()]
            # The next search starts after the end tag
            param_block_end_pos = param_close_tag_match.end()
        else:
            # Incomplete parameter block: content extends to the start of the next sibling parameter tag,
            # or to the end of the current tool block. This is another part of the "attempt to fix" logic.
            next_param_open_tag = re.search(
                r"<\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*>", tool_block_content[param_content_start:], re.DOTALL
            )

            if next_param_open_tag:
                # Content extends to the start of the next parameter tag
                param_value_raw = tool_block_content[
                    param_content_start : param_content_start + next_param_open_tag.start()
                ]
                param_block_end_pos = param_content_start + next_param_open_tag.start()
            else:
                # Content extends to the end of the current tool block
                param_value_raw = tool_block_content[param_content_start:]
                param_block_end_pos = len(tool_block_content)

        # Strip whitespace from the value and store it
        current_params[param_name] = param_value_raw.strip()
        temp_idx = param_block_end_pos

    return current_params


def parse_tool_calls(message: str) -> list[dict[str, Any]]:
    """
    Parse tool calls from a message string, with robust handling of incomplete blocks.

    If a block cannot be repaired, a ToolCallParsingError is raised.

    Args:
        message (str): The message string containing tool calls.

    Returns:
        list: A list of parsed tool call dictionaries. Each dictionary contains
              'tool_name' and 'parameters'. 'parameters' is a dictionary
              where keys are parameter names and values are parameter contents (strings).

    Examples:
        >>> parse_tool_calls("<tool1><param1>value1</param1></tool1>")
        [{'tool_name': 'tool1', 'parameters': {'param1': 'value1'}}]

        >>> parse_tool_calls("<tool2><paramA>incomplete<paramB>complete</paramB></tool2>")
        [{'tool_name': 'tool2', 'parameters': {'paramA': 'incomplete', 'paramB': 'complete'}}]

        >>> parse_tool_calls("<tool3><paramC>valueC")  # Missing closing tool tag
        [{'tool_name': 'tool3', 'parameters': {'paramC': 'valueC'}}]

    """
    tool_calls: list[dict[str, Any]] = []

    idx = 0
    while idx < len(message):
        # Find the start tag of a top-level tool call
        # Match tags of the format <tag_name>
        open_tag_match = re.match(r"<\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*>", message[idx:], re.DOTALL)
        if not open_tag_match:
            # If no start tag is found at the current position, skip one character and continue searching
            idx += 1
            continue

        tool_name = open_tag_match.group(1)
        # Get the position after the start tag ends
        tool_start_tag_end = idx + open_tag_match.end()

        # Try to find the matching end tag for the current tool
        # Use re.escape() to handle special characters in the tool name correctly
        tool_close_tag_pattern = re.compile(r"<\/\s*" + re.escape(tool_name) + r"\s*>", re.DOTALL)
        tool_close_tag_match = tool_close_tag_pattern.search(message, tool_start_tag_end)

        tool_block_content = ""
        # By default, if the tool block is incomplete, its content extends to the end of the message
        tool_block_end_pos = len(message)

        if tool_close_tag_match:
            # If a matching end tag is found, the content is between the start and end tags
            tool_block_content = message[tool_start_tag_end : tool_close_tag_match.start()]
            # The next search starts after the end tag
            tool_block_end_pos = tool_close_tag_match.end()
        else:
            # Incomplete tool block: content extends to the start of the next possible top-level tool tag,
            # or to the end of the message. This is part of the "attempt to fix" logic.
            next_tool_open_tag = re.search(r"<\s*[a-zA-Z_][a-zA-Z0-9_]*\s*>", message[tool_start_tag_end:], re.DOTALL)
            if next_tool_open_tag:
                # Content extends to the start of the next top-level tag
                tool_block_content = message[tool_start_tag_end : tool_start_tag_end + next_tool_open_tag.start()]
                tool_block_end_pos = tool_start_tag_end + next_tool_open_tag.start()
            else:
                # Content extends to the end of the message
                tool_block_content = message[tool_start_tag_end:]
                tool_block_end_pos = len(message)

        # Parse parameters from the tool block content
        current_params = _parse_params(tool_block_content, tool_name)

        # Add the parsed tool call to the results list
        tool_calls.append({"tool_name": tool_name, "parameters": current_params})
        # Update the main loop's index to continue searching for the next tool call
        idx = tool_block_end_pos

    return tool_calls
