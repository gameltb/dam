"""Pydantic models for the new session and workspace architecture."""

from __future__ import annotations

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
    content: list[ContentBlock]
    workspace_version_id: str | None = Field(
        None, description="The version of the workspace after this message's tool calls were executed."
    )


class Session(BaseModel):
    """Represents a single, independent conversation."""

    session_id: str = Field(..., description="A unique identifier for the session.")
    messages: list[Message] = Field(default_factory=list)


class Resource(BaseModel):
    """A base model for a resource within a workspace."""

    resource_id: str = Field(..., description="A unique identifier for the resource instance.")
    type: str = Field(..., description="The type of the resource (e.g., 'docker_sandbox', 'git_repo').")
    config: dict[str, Any] = Field(default_factory=dict)


class Workspace(BaseModel):
    """Represents a container for versioned resources."""

    workspace_id: str = Field(..., description="A unique identifier for the workspace.")
    resources: dict[str, Resource] = Field(default_factory=dict)


Session.model_rebuild()
