import io
import json
import os
import re
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from domarkx.utils.markdown_utils import CodeBlock

try:
    import frontmatter
except ImportError:
    frontmatter = None


@dataclass
class SessionMetadata:
    code_blocks: List[CodeBlock] = field(default_factory=list)
    session_config: Optional[dict] = None


@dataclass
class Message:
    speaker: str
    code_blocks: List[CodeBlock] = field(default_factory=list)
    blockquote: Optional[str] = None
    metadata: dict = field(default_factory=lambda: {})


@dataclass
class ParsedDocument:
    global_metadata: dict = field(default_factory=lambda: {})
    code_blocks: List[CodeBlock] = field(default_factory=list)
    config: SessionMetadata = field(default_factory=SessionMetadata)
    conversation: List[Message] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list, repr=False)


import logging


class MarkdownLLMParser:
    def __init__(self):
        self.document = ParsedDocument()
        self.state = "start"
        self.logger = logging.getLogger(__name__)
        self.source_path = None

    def _validate_message_content(self, code_blocks: List[CodeBlock], blockquote: Optional[str], speaker: str):
        if not code_blocks and (blockquote is None or blockquote.strip() == ""):
            raise ValueError(f"Section '{speaker}' must have at least one code block or a non-empty blockquote. (file: {self.source_path})")

    def parse(self, md_content: str, source_path: str = ".") -> ParsedDocument:
        self.document = ParsedDocument()
        self.state = "start"
        self.source_path = source_path

        lines = md_content.splitlines(keepends=True)
        self.logger.debug(f"Parsing {len(lines)} lines")

        i = 0
        # Parse YAML frontmatter
        if lines and lines[0].startswith("---"):
            i = 1
            yaml_lines = []
            while i < len(lines) and not lines[i].startswith("---"):
                yaml_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].startswith("---"):
                try:
                    self.document.global_metadata = yaml.safe_load("".join(yaml_lines))
                    i += 1
                    self.logger.debug(f"Parsed frontmatter: {self.document.global_metadata}")
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML front matter: {e}")

        # Parse document-level code blocks before first message
        while i < len(lines) and not lines[i].startswith("## "):
            if lines[i].startswith("```"):
                _, code_block = self._parse_code_block(lines, i)
                self.document.code_blocks.append(code_block)
                # Move i to after code block
                while i < len(lines) and not lines[i].startswith("```"):
                    i += 1
                i += 1
            else:
                i += 1

        self.document.raw_lines = lines
        self._parse_lines(lines, i)
        return self.document

    def _parse_lines(self, lines: List[str], start_index: int = 0):
        i = start_index
        while i < len(lines):
            if lines[i].startswith("## "):
                speaker = lines[i][3:].strip()
                i += 1
                code_blocks = []
                blockquote = None
                metadata = {}
                # Parse all code blocks and at most one blockquote before next message or EOF
                blockquote_found = False
                while i < len(lines) and not lines[i].startswith("## "):
                    if lines[i].startswith("```json msg-metadata"):
                        _, metadata = self._parse_metadata_block(lines, i)
                        while i < len(lines) and not lines[i].startswith("```"):
                            i += 1
                        i += 1
                    elif lines[i].startswith("```"):
                        _, code_block = self._parse_code_block(lines, i)
                        code_blocks.append(code_block)
                        while i < len(lines) and not lines[i].startswith("```"):
                            i += 1
                        i += 1
                    elif lines[i].startswith(">"):
                        if blockquote_found:
                            raise ValueError(f"Section '{speaker}' has more than one blockquote. (file: {self.source_path})")
                        _, blockquote = self._parse_blockquote(lines, i)
                        blockquote_found = True
                        # Move i to after all consecutive blockquote lines
                        while i < len(lines) and lines[i].startswith(">"):
                            i += 1
                    elif lines[i].strip() == "":
                        i += 1
                    else:
                        raise ValueError(f"Section '{speaker}' invalid content at line {i + 1}: '{lines[i].strip()}' (file: {self.source_path})")
                self._validate_message_content(code_blocks, blockquote, speaker)
                self.document.conversation.append(Message(speaker=speaker, code_blocks=code_blocks, blockquote=blockquote, metadata=metadata))
            else:
                i += 1


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
            raise ValueError(f"Error parsing msg-metadata JSON: {e}")

    def _parse_blockquote(self, lines: List[str], start_index: int) -> Tuple[int, str]:
        i = start_index
        content_lines = []
        while i < len(lines) and lines[i].startswith(">"):
            # Support multi-line blockquote, preserve line breaks
            content_lines.append(lines[i][1:].lstrip())
            i += 1
        return i, "\n".join(content_lines)

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
