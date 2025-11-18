"""For expanding macros."""

import pathlib
from typing import Any

from domarkx.utils.markdown_utils import Macro, find_first_macro


class MacroExpander:
    """Expands macros in a string."""

    def __init__(self) -> None:
        """Initialize the MacroExpander."""
        self.macros = {
            "include": self._include_macro,
            "set": self._set_macro,
        }

    def expand(
        self,
        content: str,
        base_dir: pathlib.Path,
        override_parameters: dict[str, Any] | None = None,
    ) -> str:
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
                macro_value = handler(macro, base_dir)

            # Recursively expand macros in the replacement value
            macro_value_str = self.expand(macro_value, base_dir, override_parameters)
            expanded_content = (
                expanded_content[: macro.start + expande_pos]
                + macro_value_str
                + expanded_content[macro.end + expande_pos :]
            )
            expande_pos = expande_pos + macro.start + len(macro_value_str)
        return expanded_content

    def _include_macro(self, macro: Macro, base_dir: pathlib.Path) -> str:
        """Handle the @include macro."""
        path = macro.params.get("path")
        if not path:
            return ""

        include_path = pathlib.Path(str(path))
        if not include_path.is_absolute():
            include_path = base_dir / include_path

        if include_path.exists():
            return include_path.read_text()
        # If the path does not exist, return the original macro text to avoid breaking the content.
        return ""

    def _set_macro(self, macro: Macro, _: pathlib.Path) -> str:
        """Handle the @set macro."""
        return str(macro.params.get("value", ""))
