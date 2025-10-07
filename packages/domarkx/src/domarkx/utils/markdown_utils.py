"""Markdown parsing utilities."""

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlparse


@dataclass
class CodeBlock:
    """Represents a code block extracted from Markdown."""

    language: str | None = None
    code: str = ""
    attrs: str | None = None


CODE_BLOCK_REGEX = re.compile(r"```(\w*)(?:\s*name=([\S]+))?\n(.*?)\n```", re.DOTALL)


def find_code_blocks(text: str) -> list[CodeBlock]:
    """Find all code blocks in a Markdown string."""
    matches = CODE_BLOCK_REGEX.finditer(text)
    results: list[CodeBlock] = []
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
    params: dict[str, Any] = field(default_factory=lambda: {})
    link_text: str = ""
    url: str = ""
    original_text: str = ""
    start: int = 0
    end: int = 0


MACRO_PATTERN = re.compile(r"\[@(.+?)\]\((domarkx://.+?)\)")
FOLLOWING_LINKS_PATTERN = re.compile(r"\[(.+?)\]\((.+?)\)")


def find_first_macro(content: str) -> Macro | None:
    """Find the first macro in the content and return a Macro object if found."""
    match = MACRO_PATTERN.search(content)
    if not match:
        return None

    link_text = match.group(1)
    url_string = match.group(2)
    parsed_url = urlparse(url_string)
    command = parsed_url.netloc

    # Extract params from URL query
    parsed_params = parse_qs(parsed_url.query)
    # Flatten the lists of params
    params: dict[str, str] = {k: v[0] for k, v in parsed_params.items()}

    # Check for a following URL which is treated as a parameter
    match_end = match.end()
    rest_of_content = content[match_end:]

    while True:
        following_match = FOLLOWING_LINKS_PATTERN.match(rest_of_content)
        if following_match:
            param_name = following_match.group(1)
            param_value = following_match.group(2)
            params[param_name] = param_value
            match_end += following_match.end()
            rest_of_content = content[match_end:]
        else:
            break

    original_text = content[match.start() : match_end]

    return Macro(
        command=command,
        params=params,
        link_text=link_text,
        url=url_string,
        original_text=original_text,
        start=match.start(),
        end=match_end,
    )
