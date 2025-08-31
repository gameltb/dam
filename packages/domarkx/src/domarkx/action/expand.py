import logging
import pathlib
from typing import Annotated

import typer

from domarkx.macro_expander import MacroExpander

logger = logging.getLogger(__name__)


from typing import Any, Optional


def expand(
    input_file: Annotated[
        pathlib.Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    ],
    output_file: Annotated[
        Optional[pathlib.Path],
        typer.Option("--output", "-o", help="Output file path. If not provided, defaults to <input_file>.expanded.md"),
    ] = None,
) -> None:
    """Expands macros in a Markdown document."""
    input_path = input_file
    content = input_path.read_text()

    expander = MacroExpander(base_dir=str(input_path.parent))
    expanded_content = expander.expand(content)

    if output_file:
        output_path = output_file
    else:
        output_path = input_path.with_suffix(".expanded.md")

    output_path.write_text(expanded_content)
    logger.info(f"Expanded document written to {output_path}")


def register(main_app: typer.Typer, settings: Any) -> None:
    main_app.command()(expand)
