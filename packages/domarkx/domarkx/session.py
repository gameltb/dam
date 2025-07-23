import pathlib
from abc import ABC, abstractmethod

from domarkx.utils.chat_doc_parser import MarkdownLLMParser


class Session(ABC):
    def __init__(self, doc_path: pathlib.Path):
        self.doc_path = doc_path
        self.doc = self._parse_document()
        self.agent = None
        self.tool_executors = []

    def _parse_document(self):
        with self.doc_path.open() as f:
            md_content = f.read()

        parser = MarkdownLLMParser()
        try:
            return parser.parse(md_content)
        except ValueError as e:
            raise ValueError(f"Error parsing document at {self.doc_path.absolute()}: {e}") from e

    @abstractmethod
    async def setup(self, **kwargs):
        pass

    def get_code_block(self, name: str):
        for block in self.doc.code_blocks:
            # The name of the code block is in the `attrs` field.
            if block.attrs == name:
                return block.code
        return None
