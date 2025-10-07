"""Base class for managing interactive sessions within a domarkx document."""

import pathlib
from abc import ABC, abstractmethod
from typing import Any

from domarkx.utils.chat_doc_parser import MarkdownLLMParser, ParsedDocument


class Session(ABC):
    """Abstract base class for a session."""

    def __init__(self, doc_path: pathlib.Path) -> None:
        """
        Initialize the session.

        Args:
            doc_path (pathlib.Path): The path to the document.

        """
        self.doc_path = doc_path
        self.doc = self._parse_document()
        self.agent: Any = None
        self.tool_executors: list[Any] = []

    def _parse_document(self) -> ParsedDocument:
        with self.doc_path.open() as f:
            md_content = f.read()

        parser = MarkdownLLMParser()
        try:
            return parser.parse(md_content)
        except ValueError as e:
            raise ValueError(f"Error parsing document at {self.doc_path.absolute()}: {e}") from e

    @abstractmethod
    async def setup(self, **kwargs: Any) -> None:
        """Set up the session."""
        pass

    def get_code_block(self, name: str) -> str | None:
        """
        Get the content of a code block by its name.

        Args:
            name (str): The name of the code block.

        Returns:
            str | None: The content of the code block, or None if not found.

        """
        for block in self.doc.code_blocks:
            # The name of the code block is in the `attrs` field.
            if block.attrs == name:
                return block.code
        return None
