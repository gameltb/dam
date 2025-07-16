from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import re
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class CodeBlock:
    """Represents a code block extracted from Markdown."""

    language: Optional[str] = None
    code: str = ""
    attrs: Optional[str] = None


CODE_BLOCK_REGEX = re.compile(r"```(\w*)(?:\s*name=([\S]+))?\n(.*?)\n```", re.DOTALL)


def find_code_blocks(text: str) -> List[CodeBlock]:
    """Finds all code blocks in a Markdown string."""
    matches = CODE_BLOCK_REGEX.finditer(text)
    results = []
    for match in matches:
        results.append(
            CodeBlock(
                language=match.group(1) or None,
                attrs=match.group(2) or None,
                code=match.group(3) + "\n",
            )
        )
    return results


@dataclass
class Macro:
    """Represents a macro command parsed from a Markdown link."""

    command: str
    params: Dict[str, Any] = field(default_factory=dict)
    link_text: str = ""
    url: str = ""


MACRO_REGEX = re.compile(r"\[@([a-zA-Z0-9_]+)\]\(domarkx://([a-zA-Z0-9_]+)(\?[^)]*)?\)")


def find_macros(text: str) -> List[Macro]:
    """Finds all domarkx macros in the text using regex and returns a list of Macro objects."""
    macros = []
    for match in MACRO_REGEX.finditer(text):
        macro_text = match.group(1)
        macro_name = match.group(2)
        params_str = match.group(3)
        params = {}
        if params_str:
            # Remove leading '?', split by '&', then '='
            for pair in params_str[1:].split("&"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    params[k] = v
        macros.append(
            Macro(
                command=macro_name,
                params=params,
                link_text=macro_text,
                url=f"domarkx://{macro_name}{params_str or ''}",
            )
        )
    return macros


def find_first_macro(text: str):
    """Finds the first domarkx macro in the text and returns a regex match object, or None if not found."""
    return MACRO_REGEX.search(text)
