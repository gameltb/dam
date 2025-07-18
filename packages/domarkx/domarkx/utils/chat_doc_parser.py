import io
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import yaml

from domarkx.utils.markdown_utils import CodeBlock


@dataclass
class SessionMetadata:
    code_blocks: List[CodeBlock] = field(default_factory=list)
    session_config: Optional[dict] = None


@dataclass
class Message:
    speaker: str
    code_blocks: List[CodeBlock] = field(default_factory=list)
    content: Optional[str] = None

    def get_code_blocks(self, language: str = None, attrs: str = None) -> List[CodeBlock]:
        return [
            cb
            for cb in self.code_blocks
            if (language is None or cb.language == language) and (attrs is None or cb.attrs == attrs)
        ]

    @property
    def metadata(self) -> Optional[dict]:
        blocks = self.get_code_blocks(attrs="msg-metadata")
        if not blocks:
            return None
        return json.loads(blocks[0].code)


@dataclass
class ParsedDocument:
    global_metadata: dict = field(default_factory=lambda: {})
    code_blocks: List[CodeBlock] = field(default_factory=list)
    conversation: List[Message] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list, repr=False)

    def get_code_blocks(self, language: str = None, attrs: str = None) -> List[CodeBlock]:
        return [
            cb
            for cb in self.code_blocks
            if (language is None or cb.language == language) and (attrs is None or cb.attrs == attrs)
        ]

    @property
    def session_config(self) -> Optional[dict]:
        blocks = self.get_code_blocks(attrs="session-config")
        if not blocks:
            return None
        return json.loads(blocks[0].code)


import logging


class MarkdownLLMParser:
    def __init__(self):
        self.document = ParsedDocument()
        self.logger = logging.getLogger(__name__)
        self.source_path = None

    def _validate_message_content(self, code_blocks: List[CodeBlock], content: Optional[str], speaker: str):
        if not code_blocks and (content is None or not content.strip()):
            raise ValueError(
                f"Section '{speaker}' must have at least one code block or a non-empty content. (file: {self.source_path})"
            )

    def parse(self, md_content: str, source_path: str = ".") -> ParsedDocument:
        self.document = ParsedDocument()
        self.source_path = source_path
        lines = md_content.splitlines(keepends=True)
        self.document.raw_lines = lines
        i = 0

        # Parse frontmatter
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
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML front matter: {e}")

        # Parse document-level blocks
        i = self._parse_blocks(lines, i, self.document)

        # Parse conversation
        self._parse_conversation(lines, i)

        return self.document

    def _parse_blocks(self, lines: List[str], start_index: int, target: Union[ParsedDocument, Message]):
        i = start_index
        seen_code_blocks = set()

        while i < len(lines) and not lines[i].startswith("## "):
            if lines[i].startswith("```"):
                i, code_block = self._parse_code_block(lines, i)
                signature = (code_block.language, code_block.attrs)
                if signature in seen_code_blocks:
                    raise ValueError(
                        f"Duplicate code block with language '{signature[0]}' and attrs '{signature[1]}' found. (file: {self.source_path})"
                    )
                seen_code_blocks.add(signature)
                if isinstance(target, (ParsedDocument, Message)):
                    target.code_blocks.append(code_block)

            elif lines[i].startswith(">"):
                if isinstance(target, Message):
                    if target.content is not None:
                        raise ValueError(f"Duplicate blockquote found in message. (file: {self.source_path})")
                    i, target.content = self._parse_blockquote(lines, i)
                else:
                    i += 1
            elif lines[i].strip() == "":
                i += 1
            else:
                raise ValueError(f"Invalid content at line {i + 1}: '{lines[i].strip()}' (file: {self.source_path})")

        return i

    def _parse_conversation(self, lines: List[str], start_index: int):
        i = start_index
        while i < len(lines):
            if lines[i].startswith("## "):
                speaker = lines[i][3:].strip()
                i += 1
                message = Message(speaker=speaker)
                i = self._parse_blocks(lines, i, message)
                self._validate_message_content(message.code_blocks, message.content, speaker)
                self.document.conversation.append(message)
            else:
                i += 1

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

    def _parse_blockquote(self, lines: List[str], start_index: int) -> Tuple[int, str]:
        i = start_index
        content_lines = []
        while i < len(lines) and lines[i].startswith(">"):
            content_lines.append(lines[i][1:].lstrip())
            i += 1
        return i, "".join(content_lines)


def append_message(writer: io.StringIO, message: Message):
    writer.write(f"\n## {message.speaker}\n\n")
    if message.metadata:
        writer.write(f"```json msg-metadata\n{json.dumps(message.metadata, indent=2, ensure_ascii=False)}\n```\n\n")
    if message.content:
        writer.write(f"> {message.content}\n")
    for code_block in message.code_blocks:
        writer.write(
            f"```{code_block.language or ''}{' ' + code_block.attrs if code_block.attrs else ''}\n{code_block.code}\n```\n\n"
        )
