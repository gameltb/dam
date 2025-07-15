import pathlib
import typer
import logging
from typing import Annotated

from domarkx.utils.markdown_utils import find_macros

logger = logging.getLogger(__name__)


def expand(
    input_file: Annotated[
        pathlib.Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    ],
    output_file: Annotated[
        pathlib.Path,
        typer.Option("--output", "-o", help="Output file path. If not provided, defaults to <input_file>.expanded.md"),
    ] = None,
):
    """Expands macros in a Markdown document."""
    input_path = input_file
    content = input_path.read_text()
    macros = find_macros(content)

    for macro in macros:
        if macro.command == "include":
            include_path = pathlib.Path(macro.params.get("path"))
            if not include_path.is_absolute():
                include_path = input_path.parent / include_path

            if include_path.exists():
                include_content = include_path.read_text()
                content = content.replace(f"[@{macro.link_text}]({macro.url})", include_content)
            else:
                logger.warning(f"File not found for include macro: {include_path}")

    if output_file:
        output_path = output_file
    else:
        output_path = input_path.with_suffix(".expanded.md")

    output_path.write_text(content)
    logger.info(f"Expanded document written to {output_path}")


def register(main_app: typer.Typer, settings):
    main_app.command()(expand)
