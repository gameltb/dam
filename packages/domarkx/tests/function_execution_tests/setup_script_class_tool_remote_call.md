---
title: "Setup Script Class Tool Remote Call"
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
from domarkx.tools.tool_wrapper import ToolWrapper
from domarkx.tool_executors.jupyter import JupyterToolExecutor
from autogen_core.code_executor import LocalCodeExecutor

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

code_executor = LocalCodeExecutor()
jupyter_executor = JupyterToolExecutor(code_executor=code_executor)
my_tool = MyTool()
add_tool_wrapped = ToolWrapper(tool_func=my_tool.add, executor=jupyter_executor)

tools = [add_tool_wrapped]
tool_executors = [jupyter_executor]
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
