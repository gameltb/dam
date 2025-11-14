"""Data structures for a domarkx Session Snapshot."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """A text block in a message."""

    type: Literal["text"] = "text"
    value: str


class CodeBlock(BaseModel):
    """A code block in a message."""

    type: Literal["code"] = "code"
    language: str
    name: str | None = None
    code: str


ContentBlock = TextBlock | CodeBlock


class Message(BaseModel):
    """A message in the conversation."""

    role: Literal["user", "assistant"]
    message_metadata: dict[str, Any] | None = None
    content: list[ContentBlock]


class SetupScript(BaseModel):
    """The setup script for the session."""

    language: str
    code: str


class ExecutionConfig(BaseModel):
    """The execution configuration for the session."""

    engine: str
    timeout: int = 60


class Metadata(BaseModel):
    """Metadata for the session snapshot."""

    source_file: str | None = None
    import_timestamp: str
    user_metadata: dict[str, Any] | None = None


class SessionSnapshot(BaseModel):
    """The root of the Session Snapshot."""

    version: str = "1.0"
    snapshot_id: str = Field(..., description="A unique identifier for the snapshot.")
    metadata: Metadata
    execution_config: ExecutionConfig
    setup_script: SetupScript | None = None
    conversation: list[Message]
