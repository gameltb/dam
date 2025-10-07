"""Expand macros in a Markdown document."""
import logging
import pathlib
from typing import Annotated, Any

import typer

from domarkx.macro_expander import MacroExpander

logger = logging.getLogger(__name__)


def expand(
    input_file: Annotated[
        pathlib.Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    ],
    output_file: Annotated[
        pathlib.Path | None,
        typer.Option("--output", "-o", help="Output file path. If not provided, defaults to <input_file>.expanded.md"),
    ] = None,
) -> None:
    """Expand macros in a Markdown document."""
    input_path = input_file
    content = input_path.read_text()

    expander = MacroExpander(base_dir=str(input_path.parent))
    expanded_content = expander.expand(content)

    output_path = output_file or input_path.with_suffix(".expanded.md")

    output_path.write_text(expanded_content)
    logger.info("Expanded document written to %s", output_path)


def register(main_app: typer.Typer, _: Any) -> None:
    """
    Register the `expand` command with the Typer application.

    Args:
        main_app (typer.Typer): The Typer application to register the command with.
        _ (Any): The application settings (unused).

    """
    main_app.command()(expand)
