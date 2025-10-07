"""Logging utilities for tracking LLM generations."""

import json
import logging
import pathlib
from datetime import datetime

from autogen_core import EVENT_LOGGER_NAME
from autogen_core.logging import LLMCallEvent


class LLMJsonlTracker(logging.Handler):
    """A logging handler that tracks LLM generations and logs them to a JSONL file."""

    def __init__(self, log_file: str) -> None:
        """
        Initialize the logging handler.

        Args:
            log_file (str): The path to the JSONL log file.

        """
        super().__init__()
        self.log_file = log_file

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit the log record.

        This method is called by the logging module to process a log record.

        Args:
            record (logging.LogRecord): The log record to emit.

        """
        try:
            # Use the StructuredMessage if the message is an instance of it
            if isinstance(record.msg, LLMCallEvent):
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "data": record.msg.kwargs,
                }
                with pathlib.Path(self.log_file).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            self.handleError(record)


def setup_jsonl_logger(log_file: str = "autogen_llm_generations.log.jsonl") -> None:
    """
    Set up the JSONL logger.

    Args:
        log_file (str): The path to the JSONL log file.

    """
    logger = logging.getLogger(EVENT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    llm_usage = LLMJsonlTracker(log_file)
    logger.handlers = [llm_usage]


setup_jsonl_logger()
