"""The command-line interface for domarkx."""

import importlib
import logging
import pathlib
from typing import Any

import typer
from dotenv import load_dotenv

from domarkx.config import settings
from domarkx.utils.no_border_rich_tracebacks import NoBorderRichHandler

# Configure logging
logger = logging.getLogger("domarkx")


class StrMsgOnlyFilter(logging.Filter):
    """A logging filter that only allows string messages or exceptions to be logged."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter out log records that are not strings or exceptions.

        Args:
            record (logging.LogRecord): The log record to filter.

        Returns:
            bool: True if the record should be logged, False otherwise.

        """
        return isinstance(record.msg, (str, Exception))


logger_handler = NoBorderRichHandler(
    rich_tracebacks=True,
    tracebacks_max_frames=1,
    tracebacks_show_locals=True,
    markup=False,
    show_time=False,
    show_level=True,
    show_path=True,
)
logger_handler.addFilter(StrMsgOnlyFilter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[logger_handler],
    format="[%(name)s] %(message)s",
)


cli_app = typer.Typer()


def load_actions(settings: Any) -> None:
    """
    Dynamically load and register actions from the 'cli' directory.

    Args:
        settings (Any): The application settings.

    """
    actions_dir = pathlib.Path(__file__).parent / "cli"
    for file_path in actions_dir.iterdir():
        if file_path.suffix == ".py" and not file_path.name.startswith("__"):
            module_name = file_path.stem
            module = importlib.import_module(f"domarkx.cli.{module_name}")
            if hasattr(module, "register"):
                module.register(cli_app, settings)


def main() -> None:
    """Run the main entry point for the domarkx CLI."""
    load_dotenv()
    load_actions(settings)
    try:
        cli_app()
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
