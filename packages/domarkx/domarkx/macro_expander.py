import os
import pathlib
from domarkx.utils.markdown_utils import find_macros


class MacroExpander:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.macros = {
            "include": self._include_macro,
        }

    def expand(self, content: str, parameters: dict = None) -> str:
        """Expands all macros in the given content."""
        if parameters is None:
            parameters = {}

        expanded_content = content
        macros = find_macros(content)
        for macro in macros:
            if macro.command in self.macros:
                expanded_content = self.macros[macro.command](macro, expanded_content)
            else:
                # Handle parameter expansion for other macros
                macro_content = ""
                if macro.command in parameters:
                    macro_content = parameters[macro.command]

                # Use a more robust replacement method that considers the macro's position
                expanded_content = expanded_content.replace(f"[@{macro.link_text}]({macro.url})", macro_content, 1)
        return expanded_content

    def _include_macro(self, macro, content):
        """Handles the @include macro."""
        path = macro.params.get("path")
        if not path:
            return content

        include_path = pathlib.Path(path)
        if not include_path.is_absolute():
            include_path = pathlib.Path(self.base_dir) / include_path

        if include_path.exists():
            include_content = include_path.read_text()
            return content.replace(f"[@{macro.link_text}]({macro.url})", include_content, 1)
        else:
            return content
