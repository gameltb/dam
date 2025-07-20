---
title: "Project Manager"
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
        "function_calling": True,
        "json_output": False,
        "family": "r1",
        "structured_output": False,
    },
)

from domarkx.tools.session_management import (
    create_session,
    send_message,
    get_messages,
    list_sessions,
)

tools = [create_session, send_message, get_messages, list_sessions]
```

## system_message

> You are a Project Manager AI. Your primary role is to break down user requests into manageable tasks and delegate them to other AI agents.
>
> **Task**: Your main task is to understand the user's request and create a plan to fulfill it. This often involves creating new sessions (sub-tasks) and communicating with them.
>
> **Goal**: The ultimate goal is to complete the user's request by coordinating the work of other AI agents.
>
> **Environment**: You are in a `domarkx` project. A `domarkx` project is a directory containing a `domarkx.yaml` file, which defines the project configuration. The project also contains `templates` and `sessions` directories. Sessions are conversations between you and other agents, and are stored as Markdown files in the `sessions` directory. Each session is a separate conversation and has its own message history.
>
> **Context**: You have access to the following tools:
>
> - `create_session(template_name, session_name, parameters)`: Creates a new session from a template.
> - `send_message(session_name, message)`: Sends a message to a session.
> - `get_messages(session_name)`: Retrieves the message history from a session.
> - `list_sessions()`: Lists all available sessions.
>
> Use these tools to manage the project and complete the user's request.

## user

```json msg-metadata
{
  "source": "user",
  "type": "UserMessage"
}
```

> 我们有哪些会话

## assistant

```json msg-metadata
{
  "content": [
    {
      "id": "call_0_62b052c4-7558-46fa-920a-61a9488e8d91",
      "arguments": "{}",
      "name": "list_sessions"
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
      "name": "list_sessions",
      "call_id": "call_0_62b052c4-7558-46fa-920a-61a9488e8d91",
      "is_error": false
    }
  ],
  "type": "FunctionExecutionResultMessage"
}
```

## assistant

```json msg-metadata
{
  "source": "assistant",
  "type": "AssistantMessage"
}
```

> 目前没有可用的会话。如果您需要创建新的会话，请告诉我，我可以为您安排。
