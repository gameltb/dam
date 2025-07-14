import io
import json
import os
import re
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from domarkx.utils.markdown_utils import CodeBlock

try:
    import frontmatter
except ImportError:
    frontmatter = None


@dataclass
class SessionMetadata:
    session_config: dict = field(default_factory=lambda: {})
    session_setup_code: Optional[CodeBlock] = None


@dataclass
class Message:
    speaker: str
    content: str
    metadata: dict = field(default_factory=lambda: {})


@dataclass
class ParsedDocument:
    global_metadata: dict = field(default_factory=lambda: {})
    config: SessionMetadata = field(default_factory=SessionMetadata)
    conversation: List[Message] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list, repr=False)

import logging

class MarkdownLLMParser:
    def __init__(self):
        self.document = ParsedDocument()
        self.state = "start"
        self.logger = logging.getLogger(__name__)

    def parse(self, md_content: str, source_path: str = ".") -> ParsedDocument:
        self.document = ParsedDocument()
        self.state = "start"

        lines = md_content.splitlines(keepends=True)
        self.logger.debug(f"Parsing {len(lines)} lines")

        if lines and lines[0].startswith("---"):
            i = 1
            yaml_lines = []
            while i < len(lines) and not lines[i].startswith("---"):
                yaml_lines.append(lines[i])
                i += 1

            if i < len(lines) and lines[i].startswith("---"):
                try:
                    self.document.global_metadata = yaml.safe_load("".join(yaml_lines))
                    lines = lines[i+1:]
                    self.logger.debug(f"Parsed frontmatter: {self.document.global_metadata}")
                except yaml.YAMLError as e:
                    self.document.errors.append(f"Error parsing YAML front matter: {e}")
                    self.logger.error(f"Error parsing YAML front matter: {e}")

        self.document.raw_lines = lines
        self._parse_lines(lines)
        return self.document

    def _parse_lines(self, lines: List[str]):
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("```session-config"):
                i = self._parse_session_config(lines, i)
            elif line.startswith("## "):
                speaker = line[3:].strip()
                i, message = self._parse_message(lines, i + 1, speaker)
                self.document.conversation.append(message)
            else:
                i += 1

    def _parse_session_config(self, lines: List[str], start_index: int) -> int:
        i = start_index + 1
        config_str = ""
        while i < len(lines) and not lines[i].startswith("```"):
            config_str += lines[i]
            i += 1
        i += 1

        session_config = {}
        try:
            session_config = json.loads(config_str)
        except json.JSONDecodeError as e:
            self.document.errors.append(f"Error parsing session-config JSON: {e}")

        session_setup_code = None
        if i < len(lines) and lines[i].startswith("```"):
            i, session_setup_code = self._parse_code_block(lines, i)

        self.document.config = SessionMetadata(session_config=session_config, session_setup_code=session_setup_code)
        return i

    def _parse_message(self, lines: List[str], start_index: int, speaker: str) -> Tuple[int, Message]:
        metadata = {}
        content = ""
        i = start_index

        # Skip blank lines
        while i < len(lines) and not lines[i].strip():
            i += 1

        if i < len(lines) and lines[i].startswith("```json msg-metadata"):
            i, metadata = self._parse_metadata_block(lines, i)

        # Skip blank lines
        while i < len(lines) and not lines[i].strip():
            i += 1

        if i < len(lines) and lines[i].startswith(">"):
            i, content = self._parse_blockquote(lines, i)
        elif i < len(lines) and lines[i].startswith("```"):
            i, code_block = self._parse_code_block(lines, i)
            content = f"```{code_block.language}\n{code_block.code}```"

        return i, Message(speaker=speaker, metadata=metadata, content=content)

    def _parse_metadata_block(self, lines: List[str], start_index: int) -> Tuple[int, dict]:
        i = start_index + 1
        metadata_str = ""
        while i < len(lines) and not lines[i].startswith("```"):
            metadata_str += lines[i]
            i += 1
        i += 1
        try:
            return i, json.loads(metadata_str)
        except json.JSONDecodeError as e:
            self.document.errors.append(f"Error parsing msg-metadata JSON: {e}")
            return i, {}

    def _parse_blockquote(self, lines: List[str], start_index: int) -> Tuple[int, str]:
        i = start_index
        content = ""
        while i < len(lines) and lines[i].startswith(">"):
            content += lines[i][2:]
            i += 1
        return i, content

    def _parse_code_block(self, lines: List[str], start_index: int) -> Tuple[int, CodeBlock]:
        i = start_index
        match = re.match(r"```(?:\s*([\w\+\-]+))?(?:\s*([\S]+))?", lines[i])
        language = match.group(1) if match else None
        attrs = match.group(2) if match else None

        i += 1
        code = ""
        while i < len(lines) and not lines[i].startswith("```"):
            code += lines[i]
            i += 1
        i += 1
        return i, CodeBlock(language=language, attrs=attrs, code=code)

def append_message(writer: io.StringIO, message: Message):
    writer.write(
        f"""
## {message.speaker}

```json msg-metadata
{json.dumps(message.metadata, indent=2, ensure_ascii=False)}
```

> {message.content}
"""
    )
