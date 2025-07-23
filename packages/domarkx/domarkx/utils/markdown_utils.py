import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


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


MACRO_PATTERN = re.compile(r"\[@(.+?)\]\((domarkx://.+?)\)(\s*\[(.+?)\]\((.+?)\))?")


def find_first_macro(content: str):
    """Finds the first macro in the content."""
    return MACRO_PATTERN.search(content)


def parse_macro(match, content):
    """Parses a macro match object."""
    macro_text = match.group(1)
    url = urlparse(match.group(2))
    macro_name = url.netloc

    # Extract params from URL query
    parsed_params = parse_qs(url.query)
    # Flatten the lists of params
    url_params = {k: v[0] for k, v in parsed_params.items()}

    # Check for a following URL which is treated as a parameter
    match_end = match.end()
    # Check for a following URL which is treated as a parameter
    match_end = match.end()
    rest_of_content = content[match_end:]

    following_links_pattern = re.compile(r"\s*\[(.+?)\]\((.+?)\)")
    while True:
        following_match = following_links_pattern.match(rest_of_content)
        if following_match:
            param_name = following_match.group(1)
            param_value = following_match.group(2)
            url_params[param_name] = param_value
            match_end += following_match.end()
            rest_of_content = content[match_end:]
        else:
            break

    return macro_text, macro_name, url_params, content[match.start():match_end], match_end
