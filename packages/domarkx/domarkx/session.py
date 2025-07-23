import pathlib
from abc import ABC, abstractmethod

from domarkx.macro_expander import MacroExpander
from domarkx.utils.chat_doc_parser import MarkdownLLMParser


class Session(ABC):
    def __init__(self, doc_path: pathlib.Path, override_parameters: dict = None):
        self.doc_path = doc_path
        self.doc = self._parse_document(override_parameters=override_parameters)
        self.agent = None
        self.tool_executors = []

    def _parse_document(self, override_parameters: dict = None):
        with self.doc_path.open() as f:
            md_content = f.read()

        expander = MacroExpander(base_dir=str(self.doc_path.parent))
        expanded_content = expander.expand(md_content, override_parameters=override_parameters)

        parser = MarkdownLLMParser()
        try:
            return parser.parse(expanded_content)
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
