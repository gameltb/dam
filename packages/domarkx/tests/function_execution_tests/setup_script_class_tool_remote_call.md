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
from domarkx.models.openrouter import OpenRouterR1OpenAIChatCompletionClient
from domarkx.tools.tool_wrapper import ToolWrapper
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

class MyTool:
    def __init__(self):
        pass

    def add(self, a: int, b: int) -> int:
        return a + b

from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor as ShellCommandExecutor

code_executor = ShellCommandExecutor()
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
