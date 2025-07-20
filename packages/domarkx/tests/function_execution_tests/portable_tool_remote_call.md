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
from domarkx.tools.tool_registry import get_unwrapped_tool
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

from autogen_core.code_executor import CodeExecutor

class MyCodeExecutor(CodeExecutor):
    def execute_code_block(self, language: str, code: str):
        pass

code_executor = MyCodeExecutor()
jupyter_executor = JupyterToolExecutor(code_executor=code_executor)
from domarkx.tools.tool_registry import get_unwrapped_tool, _discover_and_register_tools

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
