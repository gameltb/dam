import pathlib
import re
from urllib.parse import parse_qs, urlparse

from domarkx.utils.markdown_utils import find_first_macro, parse_macro


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
        while True:
            match = find_first_macro(expanded_content)
            if not match:
                break

            macro_text, macro_name, url_params, original_macro_text, match_end = parse_macro(match, expanded_content)

            # By default, the macro value is the original markdown link
            macro_value = original_macro_text

            # Special handlers (e.g., include)
            if hasattr(self, f"_{macro_name}_macro"):
                macro_obj = type("Macro", (), {})()
                macro_obj.command = macro_name
                macro_obj.link_text = macro_text
                macro_obj.url = match.group(2)

                # Combine and overwrite params
                combined_params = url_params.copy()
                if macro_text in override_parameters:
                    combined_params.update(override_parameters[macro_text])
                macro_obj.params = combined_params

                macro_value = getattr(self, f"_{macro_name}_macro")(macro_obj, expanded_content)

            # Recursively expand macros in the replacement value
            if isinstance(macro_value, str) and macro_value != original_macro_text:
                macro_value = self.expand(macro_value, override_parameters)

            expanded_content = expanded_content[: match.start()] + str(macro_value) + expanded_content[match_end:]
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
            # We need to find the original macro in the content and replace it
            # The macro in content would be f"[@{macro.link_text}]({macro.url})"
            original_macro_text = f"[@{macro.link_text}]({macro.url})"
            return content.replace(original_macro_text, include_content, 1)
        else:
            return content
