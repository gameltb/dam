import pathlib
from typing import Any, Optional

from domarkx.utils.chat_doc_parser import MarkdownLLMParser, ParsedDocument
from domarkx.utils.markdown_utils import Macro, find_first_macro


class MacroExpander:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.macros = {
            "include": self._include_macro,
            "set": self._set_macro,
        }

    def expand(self, content: str, override_parameters: Optional[dict[str, Any]] = None) -> str:
        """Expands macros in the content sequentially: find and expand the first macro, repeat until all macros are processed."""
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
                macro_value = handler(macro, expanded_content)

            # Recursively expand macros in the replacement value
            macro_value_str = self.expand(macro_value, override_parameters)
            expanded_content = (
                expanded_content[: macro.start + expande_pos]
                + macro_value_str
                + expanded_content[macro.end + expande_pos :]
            )
            expande_pos = expande_pos + macro.start + len(macro_value_str)
        return expanded_content  # type: ignore[no-any-return]

    def _include_macro(self, macro: Macro, content: str) -> str:
        """Handles the @include macro."""
        path = macro.params.get("path")
        if not path:
            raise Exception()

        include_path = pathlib.Path(str(path))
        if not include_path.is_absolute():
            include_path = pathlib.Path(self.base_dir) / include_path

        if include_path.exists():
            return include_path.read_text()
        else:
            # If the path does not exist, return the original macro text to avoid breaking the content.
            return ""

    def _set_macro(self, macro: Macro, content: str) -> str:
        """Handles the @set macro."""
        return str(macro.params.get("value", ""))


class DocExpander:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.parser = MarkdownLLMParser()

    def expand(self, content: str) -> ParsedDocument:
        # Parse the document first
        parsed_doc = self.parser.parse(content)

        # Create a new MacroExpander with the document's directory as the base
        macro_expander = MacroExpander(self.base_dir)

        # Expand macros in each message's content
        for message in parsed_doc.conversation:
            if message.content:
                message.content = macro_expander.expand(message.content)

        return parsed_doc
