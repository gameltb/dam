"""Extracts code blocks from a domarkx document and saves them to files."""
import logging
import pathlib
import re
from typing import Annotated, Any

import rich
import rich.markdown
import typer
from prompt_toolkit import PromptSession
from rich.console import Console

from domarkx.utils.chat_doc_parser import MarkdownLLMParser

logger = logging.getLogger(__name__)


# Regex patterns to extract a filename/path from the first line of a code block's content.
# Order matters: more specific or common patterns first.
FILENAME_PATTERNS = [
    # Matches: #!/usr/bin/env python3 path/to/script.py -> path/to/script.py
    # Matches: #!/bin/bash path/to/script.sh -> path/to/script.sh
    re.compile(r"^\s*#!\s*(?:[\w\/\.-]+/env\s+\w+\s+)?([\w\/\.-]+\.[a-zA-Z0-9]+)\s*"),
    # Matches: # path/to/file.ext or # file.ext
    re.compile(r"^\s*#\s*([\w\/\.-]+\.[a-zA-Z0-9]+)\s*"),
    # Matches: /* path/to/file.css */ or /* file.tcss */
    re.compile(r"^\s*\/\*\s*([\w\/\.-]+\.[a-zA-Z0-9]+)\s*\*\/"),
    # Matches: ; alembic.ini or ; path/to/alembic.ini
    re.compile(r"^\s*;+\s*([\w\/\.-]+\.ini)\s*"),
    # For general markdown files if specified like: re.compile(r"^\s*\s*")
]


def do_extract_code_to_file(
    output_base_dir: str, block_inner_content: str, filepath_extracted: str | None = None
) -> None:
    """
    Extract code from a string and save it to a file.

    It attempts to automatically determine the filename from the first line of the code block.

    Args:
        output_base_dir (str): The base directory to save the file in.
        block_inner_content (str): The content of the code block.
        filepath_extracted (str | None): An optional pre-extracted filepath.

    """
    block_lines = block_inner_content.strip().split("\n")
    if not block_lines:
        logger.warning("Block is empty, skipping.")
        return

    first_line = block_lines[0].strip()
    # By default, the first line (comment) is NOT part of the final code,
    # unless it's a shebang or a comment type that is typically kept (like CSS block comments).
    first_line_is_code = False

    for _, pattern in enumerate(FILENAME_PATTERNS):
        match = pattern.match(first_line)
        if match:
            filepath_extracted = match.group(1).strip()
            # Shebangs must be the first line of the script.
            if first_line.startswith("#!"):
                first_line_is_code = True
            # CSS comments that define filename are usually not kept if they are just markers,
            # but sometimes block comments start a file. For simplicity, we'll treat it as not code for now
            # unless the user wants to adjust. Let's assume if it matched, it was a marker.
            # If the comment IS the path, we usually want to strip it for Python/other similar files.
            break

    filepath_extracted = PromptSession[str | None]().prompt(
        "filepath > ", default=filepath_extracted if filepath_extracted else ""
    )

    if filepath_extracted:
        # Construct full path relative to the output_base_dir
        full_path = pathlib.Path(output_base_dir) / filepath_extracted

        # Ensure directory structure exists
        dir_name = full_path.parent
        if dir_name:  # If there's a directory part (e.g., "dam/tui")
            dir_name.mkdir(parents=True, exist_ok=True)

        # Determine the actual code to write
        code_to_write = "\n".join(block_lines) if first_line_is_code else "\n".join(block_lines[1:])

        # Ensure there's content to write, especially after stripping the first line
        if not code_to_write.strip() and not first_line_is_code and len(block_lines) == 1:
            logger.warning("File '%s' would be empty after stripping comment. Skipping.", filepath_extracted)
            return
        if not code_to_write.strip() and first_line_is_code:  # e.g. only a shebang
            logger.info("Writing file '%s' which might only contain the shebang/comment.", filepath_extracted)

        try:
            with full_path.open("w", encoding="utf-8") as f:
                f.write(code_to_write)
            logger.info("Extracted and wrote: %s", full_path)
        except OSError as e:
            logger.error("Error writing file %s: %s", full_path, e)
        except Exception as e:
            logger.error("An unexpected error occurred while writing %s: %s", full_path, e)
    else:
        logger.warning('No filename pattern matched for first line: "%s..."', first_line[:70])


def extract_code_to_file(
    doc: Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    message_index: int,
    code_block_in_message_index: int,
) -> None:
    """
    Extract a specific code block from a document and save it to a file.

    Args:
        doc (pathlib.Path): The path to the document.
        message_index (int): The index of the message containing the code block.
        code_block_in_message_index (int): The index of the code block within the message.

    """
    with doc.open() as f:
        md_content = f.read()

    parser = MarkdownLLMParser()
    parsed_doc = parser.parse(md_content)
    message_obj = parsed_doc.conversation[message_index]
    code_block = message_obj.code_blocks[code_block_in_message_index]

    console = Console(markup=False)
    if message_obj.content:
        md = rich.markdown.Markdown(message_obj.content)
        console.rule("message")
        console.print(md)
    console.rule("code")
    console.print(rich.markdown.Markdown(f"```{code_block.language}\n{code_block.code}\n```"))

    do_extract_code_to_file(".", code_block.code, filepath_extracted=code_block.attrs)


def register(main_app: typer.Typer, _: Any) -> None:
    """
    Register the `extract_code_to_file` command with the Typer application.

    Args:
        main_app (typer.Typer): The Typer application to register the command with.
        _ (Any): The application settings (unused).

    """
    main_app.command()(extract_code_to_file)
