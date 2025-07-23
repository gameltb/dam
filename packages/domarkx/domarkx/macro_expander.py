import pathlib

from domarkx.utils.markdown_utils import Macro, find_first_macro


class MacroExpander:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.macros = {
            "include": self._include_macro,
        }

    def expand(self, content: str, override_parameters: dict = None) -> str:
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
            macro_value = macro.original_text

            # Special handlers (e.g., include)
            if hasattr(self, f"_{macro.command}_macro"):
                # Combine and overwrite params
                if macro.link_text in override_parameters:
                    macro.params.update(override_parameters[macro.link_text])

                macro_value = getattr(self, f"_{macro.command}_macro")(macro, expanded_content)

            # Recursively expand macros in the replacement value
            if isinstance(macro_value, str) and macro_value != macro.original_text:
                macro_value = self.expand(macro_value, override_parameters)

            expanded_content = expanded_content[: macro.start] + str(macro_value) + expanded_content[macro.end :]
            expande_pos = macro.start + len(str(macro_value))
        return expanded_content

    def _include_macro(self, macro: Macro, content: str) -> str:
        """Handles the @include macro."""
        path = macro.params.get("path")
        if not path:
            return content

        include_path = pathlib.Path(path)
        if not include_path.is_absolute():
            include_path = pathlib.Path(self.base_dir) / include_path

        if include_path.exists():
            return include_path.read_text()
        else:
            # If the path does not exist, return the original macro text to avoid breaking the content.
            return macro.original_text
