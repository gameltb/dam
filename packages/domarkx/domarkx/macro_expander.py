import pathlib

from domarkx.utils.markdown_utils import find_first_macro


class MacroExpander:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.macros = {
            "include": self._include_macro,
        }

    def expand(self, content: str, parameters: dict = None) -> str:
        """Expands macros in the content sequentially: find and expand the first macro, repeat until all macros are processed."""
        if parameters is None:
            parameters = {}

        expanded_content = content
        while True:
            match = find_first_macro(expanded_content)
            if not match:
                break
            macro_name = match.group(2)
            macro_text = match.group(1)
            macro_value = parameters.get(macro_name, parameters.get(macro_text, match.group(0)))
            # If macro_name is a special handler (e.g. include), use handler
            if hasattr(self, f"_{macro_name}_macro"):
                macro_obj = type("Macro", (), {})()
                macro_obj.command = macro_name
                macro_obj.link_text = macro_text
                macro_obj.url = f"domarkx://{macro_name}"
                macro_obj.params = {}
                macro_value = getattr(self, f"_{macro_name}_macro")(macro_obj, expanded_content)
            expanded_content = expanded_content[: match.start()] + str(macro_value) + expanded_content[match.end() :]
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
