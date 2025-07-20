---
title: "Setup Script Class Tool Local Call"
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
                            "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'},
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

class MyTool:
    def __init__(self):
        pass

    def add(self, a: int, b: int) -> int:
        return a + b

tools = [MyTool().add]
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

> Add 2 and 2.

## assistant

```json msg-metadata
{
  "content": [
    {
      "id": "call_123",
      "arguments": "{\"a\": 2, \"b\": 2}",
      "name": "add"
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
      "name": "add",
      "call_id": "call_123",
      "is_error": false
    }
  ],
  "type": "FunctionExecutionResultMessage"
}
```

> 4
