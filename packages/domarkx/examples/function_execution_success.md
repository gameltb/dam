---
title: "Function Execution Success"
---

## user

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
      "is_error": false,
      "result": 4
    }
  ],
  "type": "FunctionExecutionResultMessage"
}
```

## assistant

> The sum is 4.
