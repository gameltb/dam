import pytest

from domarkx.utils.chat_doc_parser import MarkdownLLMParser


def test_parse_message_blocks_and_code():
    md = """---\ntitle: "Test Session"\n---\n\n```json session-config\n{"type": "AssistantAgentState"}\n```
```python setup-script\nprint("setup")\n```
## User\n\n```json msg-metadata\n{"source": "user", "type": "UserMessage"}\n```
```python\nprint("hello")\n```
> User's message content.\n> Second line.\n\n## Assistant\n\n```json msg-metadata\n{"source": "assistant", "type": "AssistantMessage"}\n```
> Assistant's reply.\n> Multi-line reply.\n"""
    parser = MarkdownLLMParser()
    doc = parser.parse(md)
    assert doc.global_metadata["title"] == "Test Session"
    assert len(doc.code_blocks) == 2
    assert doc.code_blocks[0].language == "json"
    assert doc.code_blocks[0].attrs == "session-config"
    assert doc.code_blocks[1].language == "python"
    assert doc.code_blocks[1].attrs == "setup-script"
    assert len(doc.conversation) == 2
    user_msg = doc.conversation[0]
    assert user_msg.speaker == "User"
    assert len(user_msg.code_blocks) == 2
    assert user_msg.code_blocks[0].language == "json"
    assert user_msg.code_blocks[1].language == "python"
    assert user_msg.content.strip() == "User's message content.\nSecond line."
    assistant_msg = doc.conversation[1]
    assert assistant_msg.speaker == "Assistant"
    assert len(assistant_msg.code_blocks) == 1
    assert assistant_msg.content.strip() == "Assistant's reply.\nMulti-line reply."


def test_message_requires_code_or_content():
    md1 = """## User\n\n"""
    parser = MarkdownLLMParser()
    with pytest.raises(ValueError):
        parser.parse(md1)
    # Only content, no code block
    md2 = """## User\n\n> Only content\n"""
    doc = parser.parse(md2)
    assert doc.conversation[0].content.strip() == "Only content"
    assert len(doc.conversation[0].code_blocks) == 0
    # Only code block, no content
    md3 = """## User\n\n```python\nprint(123)\n```\n"""
    doc = parser.parse(md3)
    assert doc.conversation[0].content is None
    assert len(doc.conversation[0].code_blocks) == 1


def test_message_multiple_blockquotes():
    md = """## User\n\n> First blockquote\n\n> Second blockquote\n"""
    parser = MarkdownLLMParser()
    with pytest.raises(ValueError):
        parser.parse(md)


def test_to_markdown_serialization():
    md = """---
title: Test Session
---

```json session-config
{"type": "AssistantAgentState"}
```

```python setup-script
print("setup")
```

## User

```json msg-metadata
{"source": "user", "type": "UserMessage"}
```

```python
print("hello")
```

> User's message content.
> Second line.

## Assistant

```json msg-metadata
{"source": "assistant", "type": "AssistantMessage"}
```

> Assistant's reply.
> Multi-line reply.
"""
    parser = MarkdownLLMParser()
    doc = parser.parse(md)
    serialized_md = doc.to_markdown()
    # Normalize the markdown by removing leading/trailing whitespace and empty lines
    normalized_original = "\n".join(filter(None, (line.strip() for line in md.splitlines()))).strip()
    normalized_serialized = "\n".join(filter(None, (line.strip() for line in serialized_md.splitlines()))).strip()

    # Further normalization to handle subtle differences in YAML and JSON formatting
    normalized_original = normalized_original.replace("'", '"')
    normalized_serialized = normalized_serialized.replace("'", '"')

    assert normalized_serialized == normalized_original
