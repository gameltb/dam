import io
import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union, cast

import yaml

from domarkx.utils.markdown_utils import CodeBlock


@dataclass
class Message:
    speaker: str
    code_blocks: List[CodeBlock] = field(default_factory=list)
    content: Optional[str] = None

    def get_code_blocks(self, language: Optional[str] = None, attrs: Optional[str] = None) -> List[CodeBlock]:
        return [
            cb
            for cb in self.code_blocks
            if (language is None or cb.language == language) and (attrs is None or cb.attrs == attrs)
        ]

    @property
    def metadata(self) -> Optional[dict[str, Any]]:
        blocks = self.get_code_blocks(attrs="msg-metadata")
        if not blocks:
            return None
        return cast(dict[str, Any], json.loads(blocks[0].code))


@dataclass
class ParsedDocument:
    global_metadata: dict[str, Any] = field(default_factory=lambda: {})
    code_blocks: List[CodeBlock] = field(default_factory=list)
    conversation: List[Message] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Serializes the document back to a markdown string."""
        writer = io.StringIO()

        if self.global_metadata:
            writer.write("---\n")
            yaml.dump(self.global_metadata, writer)
            writer.write("---\n\n")

        for code_block in self.code_blocks:
            writer.write(
                f"```{code_block.language or ''}{' ' + code_block.attrs if code_block.attrs else ''}\n{code_block.code}\n```\n\n"
            )

        for message in self.conversation:
            append_message(writer, message)

        return writer.getvalue()

    def get_code_blocks(self, language: Optional[str] = None, attrs: Optional[str] = None) -> List[CodeBlock]:
        return [
            cb
            for cb in self.code_blocks
            if (language is None or cb.language == language) and (attrs is None or cb.attrs == attrs)
        ]

    @property
    def session_config(self) -> Optional[dict[str, Any]]:
        blocks = self.get_code_blocks(attrs="session-config")
        if not blocks:
            return None
        return cast(dict[str, Any], json.loads(blocks[0].code))


class MarkdownLLMParser:
    def __init__(self) -> None:
        self.document = ParsedDocument()
        self.logger = logging.getLogger(__name__)

    def _validate_message_content(self, code_blocks: List[CodeBlock], content: Optional[str], speaker: str) -> None:
        if not code_blocks and (content is None or not content.strip()):
            raise ValueError(f"Section '{speaker}' must have at least one code block or a non-empty content.")

    def parse(self, md_content: str) -> ParsedDocument:
        self.document = ParsedDocument()
        lines = md_content.splitlines(keepends=True)
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

    def _parse_blocks(self, lines: List[str], start_index: int, target: Union[ParsedDocument, Message]) -> int:
        i = start_index
        seen_code_blocks = set()

        while i < len(lines) and not lines[i].startswith("## "):
            if lines[i].startswith("```"):
                i, code_block = self._parse_code_block(lines, i)
                signature = (code_block.language, code_block.attrs)
                if signature in seen_code_blocks:
                    raise ValueError(
                        f"Duplicate code block with language '{signature[0]}' and attrs '{signature[1]}' found."
                    )
                seen_code_blocks.add(signature)
                if isinstance(target, (ParsedDocument, Message)):
                    target.code_blocks.append(code_block)

            elif lines[i].startswith(">"):
                if isinstance(target, Message):
                    if target.content is not None:
                        raise ValueError("Duplicate blockquote found in message.")
                    i, target.content = self._parse_blockquote(lines, i)
                else:
                    i += 1
            elif lines[i].strip() == "":
                i += 1
            else:
                raise ValueError(f"Invalid content at line {i + 1}: '{lines[i].strip()}'")

        return i

    def _parse_conversation(self, lines: List[str], start_index: int) -> None:
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


def append_message(writer: io.TextIOBase, message: Message) -> None:
    writer.write(f"\n## {message.speaker}\n\n")
    for code_block in message.code_blocks:
        writer.write(
            f"```{code_block.language or ''}{' ' + code_block.attrs if code_block.attrs else ''}\n{code_block.code}\n```\n\n"
        )
    if message.content:
        writer.write(textwrap.indent(message.content, "> ", predicate=lambda _: True))
        writer.write("\n\n")
