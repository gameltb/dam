"""Custom logger for the Sire package."""

import logging
import sys
from pathlib import Path
from typing import ClassVar, TypedDict


class LevelConfig(TypedDict):
    """Configuration for a logging level."""

    level_color: str
    message_color: str
    path_color: str
    width: int


class LoggerFormatter(logging.Formatter):
    """A custom logger formatter that adds colors to the output."""

    COLORS: ClassVar[dict[str, str]] = {
        "white": "\033[37m",
        "grey": "\033[90m",
        "blue": "\033[34m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "bold_red": "\033[31;1m",
        "cyan": "\033[36m",
        "reset": "\033[0m",
    }

    LEVEL_CONFIGS: ClassVar[dict[int, LevelConfig]] = {
        logging.DEBUG: {
            "level_color": "grey",
            "message_color": "grey",
            "path_color": "cyan",
            "width": 8,
        },
        logging.INFO: {
            "level_color": "green",
            "message_color": "white",
            "path_color": "cyan",
            "width": 8,
        },
        logging.WARNING: {
            "level_color": "yellow",
            "message_color": "white",
            "path_color": "cyan",
            "width": 8,
        },
        logging.ERROR: {
            "level_color": "red",
            "message_color": "red",
            "path_color": "cyan",
            "width": 8,
        },
        logging.CRITICAL: {
            "level_color": "bold_red",
            "message_color": "bold_red",
            "path_color": "cyan",
            "width": 8,
        },
    }

    def __init__(self, project_root: str | None = None):
        """
        Initialize the formatter.

        Args:
            project_root: The root directory of the project.

        """
        super().__init__()
        self.project_root = Path(project_root or Path.cwd())
        self._formatters: dict[int, logging.Formatter] = self._compile_formatters()

    def _compile_formatters(self) -> dict[int, logging.Formatter]:
        formatters: dict[int, logging.Formatter] = {}
        base_fmt = (
            "{grey}%(asctime)s{reset} "
            "[{level_color}%(levelname).1s{reset}] "
            "({thread_color}%(threadName)s{reset}) "
            "{path_color}%(pathname)s:%(lineno)d{reset} %(name)s - "
            "{message_color}%(message)s{reset}"
        )

        for level, config in self.LEVEL_CONFIGS.items():
            fmt = base_fmt.format(
                grey=self.COLORS["grey"],
                reset=self.COLORS["reset"],
                level_color=self.COLORS[config["level_color"]],
                message_color=self.COLORS[config["message_color"]],
                path_color=self.COLORS[config["path_color"]],
                thread_color=self.COLORS["blue"],
            )

            formatters[level] = logging.Formatter(fmt=fmt, datefmt="%y-%m-%d %H:%M:%S")
        return formatters

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        pathname = Path(record.pathname)
        if self.project_root and pathname.is_relative_to(self.project_root):
            record.pathname = str(pathname.relative_to(self.project_root))
        formatter = self._formatters.get(record.levelno, self._formatters[logging.DEBUG])
        return formatter.format(record)


def setup_logger(name: str | None = None) -> logging.Logger:
    """
    Set up a logger with the custom formatter.

    Args:
        name: The name of the logger.

    Returns:
        The configured logger.

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(LoggerFormatter())
        logger.addHandler(handler)

    return logger
