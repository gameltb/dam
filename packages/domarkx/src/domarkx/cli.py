import importlib
import logging
import os
from typing import Any

import typer
from dotenv import load_dotenv

from domarkx.config import settings
from domarkx.utils.no_border_rich_tracebacks import NoBorderRichHandler

# Configure logging
logger = logging.getLogger("domarkx")


class StrMsgOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
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
    actions_dir = os.path.join(os.path.dirname(__file__), "action")
    for filename in os.listdir(actions_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            module = importlib.import_module(f"domarkx.action.{module_name}")
            if hasattr(module, "register"):
                module.register(cli_app, settings)


def main() -> None:
    load_dotenv()
    load_actions(settings)
    try:
        cli_app()
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == "__main__":
    main()
