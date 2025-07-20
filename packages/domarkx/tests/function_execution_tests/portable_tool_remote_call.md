---
title: "Portable Tool Remote Call"
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
from domarkx.tools.tool_registry import get_unwrapped_tool, _discover_and_register_tools
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

code_executor = LocalCodeExecutor()
jupyter_executor = JupyterToolExecutor(code_executor=code_executor)

_discover_and_register_tools()
tool_execute_command_unwrapped = get_unwrapped_tool("tool_execute_command")
tool_execute_command_wrapped = ToolWrapper(tool_func=tool_execute_command_unwrapped, executor=jupyter_executor)

tools = [tool_execute_command_wrapped]
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
