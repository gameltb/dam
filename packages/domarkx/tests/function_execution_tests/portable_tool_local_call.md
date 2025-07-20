---
title: "Portable Tool Local Call"
---

```json session-config
{
  "type": "AssistantAgentState",
  "version": "1.0.0",
  "llm_context": {}
}
```

```python setup-script
from unittest.mock import MagicMock, AsyncMock
from autogen_core.models._types import CreateResult
from domarkx.tools.tool_registry import get_tool, _discover_and_register_tools

client = MagicMock()
async def mock_create_stream(*args, **kwargs):
    yield CreateResult(
        choices=[
            {
                "message": {
                    "content": "Mocked response",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {"name": "tool_execute_command", "arguments": '{"command": "ls"}'},
                            "type": "function",
                        }
                    ],
                }
            }
        ],
        cost=0,
        model="mock_model",
        extra_data={},
    )

client.create_stream = mock_create_stream

_discover_and_register_tools()
tools = [get_tool("tool_execute_command")]
```

## system_message



> You are a helpful AI assistant.

## user

```json msg-metadata
{
  "source": "user",
  "type": "UserMessage"
}
```

> list files in current directory

## assistant

```json msg-metadata
{
  "content": [
    {
      "id": "call_123",
      "arguments": "{\"command\": \"ls\"}",
      "name": "tool_execute_command"
    }
  ],
  "source": "assistant",
  "type": "AssistantMessage"
}
```

## unknow

```json msg-metadata
{
  "content": [
    {
      "name": "tool_execute_command",
      "call_id": "call_123",
      "is_error": false
    }
  ],
  "type": "FunctionExecutionResultMessage"
}
```

> README.md
> coverage.xml
> docs
> domarkx
> domarkx.egg-info
> editors
> pyproject.toml
> tests
