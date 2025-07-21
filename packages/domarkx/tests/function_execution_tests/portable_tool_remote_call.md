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
from domarkx.models.openrouter import OpenRouterR1OpenAIChatCompletionClient
from domarkx.tools.tool_factory import default_tool_factory
from domarkx.tools.tool_factory import default_tool_factory
from domarkx.tool_executors.jupyter import JupyterToolExecutor

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

from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

code_executor = LocalCommandLineCodeExecutor()
jupyter_executor = JupyterToolExecutor(code_executor=code_executor)
from domarkx.tools.portable_tools.execute_command import tool_execute_command

tool_execute_command_wrapped = default_tool_factory.wrap_function(func=tool_execute_command, executor=jupyter_executor)

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
