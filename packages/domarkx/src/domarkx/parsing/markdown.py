"""For parsing Markdown documents into sessions and workspace configurations."""

import pathlib
import re
from typing import Any

import mistune
import yaml

from domarkx.data.models import CodeBlock, Message, TextBlock


class MarkdownParser:
    """Parses a Markdown file into a conversation and a list of resource configs."""

    def parse(self, markdown_file: pathlib.Path) -> tuple[list[Message], list[dict[str, Any]]]:
        """
        Parse a Markdown file.

        Args:
            markdown_file (pathlib.Path): The path to the Markdown file.

        Returns:
            A tuple containing the list of messages and the list of resource configurations.

        """
        content = markdown_file.read_text()

        workspace_config_str = self._extract_workspace_config(content)
        workspace_config: dict[str, Any] = yaml.safe_load(workspace_config_str) if workspace_config_str else {}
        resource_configs = workspace_config.get("resources", [])

        conversation = self._parse_conversation(content)

        return conversation, resource_configs

    def _extract_workspace_config(self, content: str) -> str | None:
        """Extract the workspace config block."""
        match = re.search(r"```workspace-config\n(.*?)\n```", content, re.DOTALL)
        return match.group(1) if match else None

    def _parse_conversation(self, content: str) -> list[Message]:
        """Parse the conversation from the Markdown."""
        renderer = ConversationRenderer()
        markdown = mistune.create_markdown(renderer=renderer)
        return markdown(content)  # type: ignore


class ConversationRenderer(mistune.HTMLRenderer):
    """A mistune renderer that extracts the conversation."""

    ROLE_HEADING_LEVEL = 2

    def __init__(self) -> None:
        """Initialize the ConversationRenderer."""
        super().__init__()
        self.messages: list[Message] = []
        self._current_message: Message | None = None

    def heading(self, text: str, level: int, **_attrs: Any) -> str:
        """Process a heading."""
        if level == self.ROLE_HEADING_LEVEL:
            role = text.lower()
            if role in ("user", "assistant"):
                self._current_message = Message(role=role, content=[], workspace_version_id=None)
                self.messages.append(self._current_message)
        return ""

    def paragraph(self, text: str) -> str:
        """Process a paragraph."""
        if self._current_message:
            self._current_message.content.append(TextBlock(value=text))
        return ""

    def block_code(self, code: str, info: str | None = "") -> str:
        """Process a code block."""
        if self._current_message and info != "workspace-config":
            language = info.split(" ")[0] if info else ""
            self._current_message.content.append(CodeBlock(code=code, language=language))
        return ""

    def finalize(self, _data: Any) -> list[Message]:
        """Finalize the parsing."""
        return self.messages
