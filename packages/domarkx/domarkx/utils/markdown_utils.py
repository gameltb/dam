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

def find_code_blocks(text: str) -> List[CodeBlock]:
    """Finds all code blocks in a Markdown string."""
    pattern = re.compile(r"```(\w*)(?:\s*name=([\S]+))?\n(.*?)\n```", re.DOTALL)
    matches = pattern.finditer(text)
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

import mistune
from urllib.parse import urlparse, parse_qs

def find_macros(text: str) -> List[Macro]:
    """Finds all macro commands in a Markdown string."""
    macros = []

    class MacroRenderer(mistune.BaseRenderer):
        def __init__(self):
            super().__init__()
            self.macros = []

        def link(self, token, state):
            text = self.render_children(token, state)
            url = token['attrs']['url']
            if text and text.startswith("@") and not text.startswith("@@"):
                if url.startswith("domarkx://"):
                    parsed_url = urlparse(url)
                    command = parsed_url.hostname
                    params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parsed_url.query).items()}
                    self.macros.append(
                        Macro(
                            command=command,
                            params=params,
                            link_text=text[1:],
                            url=url,
                        )
                    )
            return text

        def render_children(self, token, state):
            children = token.get('children', [])
            return "".join([self.render_token(child, state) for child in children])

        def render_token(self, token, state):
            func_name = token['type']
            func = getattr(self, func_name, None)
            if func:
                return func(token, state)
            return self.text(token, state)

        def text(self, token, state):
            return token.get('raw', '')

        def paragraph(self, token, state):
            return self.render_children(token, state)

    renderer = MacroRenderer()
    markdown = mistune.create_markdown(renderer=renderer)
    markdown(text)
    return renderer.macros
    markdown(text)
    return macros
