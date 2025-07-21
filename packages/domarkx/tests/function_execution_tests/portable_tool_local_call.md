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
from domarkx.models.openrouter import OpenRouterR1OpenAIChatCompletionClient
from domarkx.tools.tool_factory import default_tool_factory

client = OpenRouterR1OpenAIChatCompletionClient(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="mock_key",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "r1",
        "structured_output": False,
    },
)

from domarkx.tools.portable_tools.execute_command import tool_execute_command

tools = [tool_execute_command]
tool_executors = []
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
