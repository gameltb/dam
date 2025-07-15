---
title: "[@session_name](domarkx://session_name)"
version: "0.0.1"
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
import os

client = OpenRouterR1OpenAIChatCompletionClient(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "r1",
        "structured_output": False,
    },
)
```

## system_message

> You are a helpful assistant. Your task is to respond to the user's prompt.

## Conversation

**User**: [@user_prompt](domarkx://user_prompt)
