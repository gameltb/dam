"""Expands macros in markdown files."""
import pathlib
from typing import Any

from domarkx.utils.chat_doc_parser import MarkdownLLMParser, ParsedDocument
from domarkx.utils.markdown_utils import Macro, find_first_macro


class MacroExpander:
    """Expands macros in a string."""

    def __init__(self, base_dir: str):
        """
        Initialize the MacroExpander.

        Args:
            base_dir (str): The base directory for resolving relative paths in macros.

        """
        self.base_dir = base_dir
        self.macros = {
            "include": self._include_macro,
            "set": self._set_macro,
        }

    def expand(self, content: str, override_parameters: dict[str, Any] | None = None) -> str:
        """
        Expand macros in the content sequentially.

        It finds and expands the first macro, then repeats until all macros are processed.
        """
        if override_parameters is None:
            override_parameters = {}

        expanded_content = content
        expande_pos = 0
        while True:
            macro = find_first_macro(expanded_content[expande_pos:])
            if not macro:
                break

            # By default, the macro value is the original markdown link
            macro_value: str = ""

            # Special handlers (e.g., include)
            if macro.command in self.macros:
                # Combine and overwrite params
                if macro.link_text in override_parameters:
                    macro.params.update(override_parameters[macro.link_text])

                handler = self.macros[macro.command]
                macro_value = handler(macro)

            # Recursively expand macros in the replacement value
            macro_value_str = self.expand(macro_value, override_parameters)
            expanded_content = (
                expanded_content[: macro.start + expande_pos]
                + macro_value_str
                + expanded_content[macro.end + expande_pos :]
            )
            expande_pos = expande_pos + macro.start + len(macro_value_str)
        return expanded_content  # type: ignore[no-any-return]

    def _include_macro(self, macro: Macro) -> str:
        """Handle the @include macro."""
        path = macro.params.get("path")
        if not path:
            raise Exception()

        include_path = pathlib.Path(str(path))
        if not include_path.is_absolute():
            include_path = pathlib.Path(self.base_dir) / include_path

        if include_path.exists():
            return include_path.read_text()
        # If the path does not exist, return the original macro text to avoid breaking the content.
        return ""

    def _set_macro(self, macro: Macro) -> str:
        """Handle the @set macro."""
        return str(macro.params.get("value", ""))


class DocExpander:
    """Expands macros in a parsed document."""

    def __init__(self, base_dir: str):
        """
        Initialize the DocExpander.

        Args:
            base_dir (str): The base directory for resolving relative paths.

        """
        self.base_dir = base_dir
        self.parser = MarkdownLLMParser()

    def expand(self, content: str) -> ParsedDocument:
        """
        Expand macros in the content of a parsed document.

        Args:
            content (str): The markdown content to expand.

        Returns:
            ParsedDocument: The document with expanded macros.

        """
        # Parse the document first
        parsed_doc = self.parser.parse(content)

        # Create a new MacroExpander with the document's directory as the base
        macro_expander = MacroExpander(self.base_dir)

        # Expand macros in each message's content
        for message in parsed_doc.conversation:
            if message.content:
                message.content = macro_expander.expand(message.content)

        return parsed_doc
