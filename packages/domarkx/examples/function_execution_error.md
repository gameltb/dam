---
title: "Function Execution Error"
---

## user

> Divide by zero.

## assistant

```json msg-metadata
{
  "content": [
    {
      "id": "call_456",
      "arguments": "{\"a\": 1, \"b\": 0}",
      "name": "divide"
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
      "name": "divide",
      "call_id": "call_456",
      "is_error": true,
      "error": "ZeroDivisionError: division by zero"
    }
  ],
  "type": "FunctionExecutionResultMessage"
}
```

## assistant

> I'm sorry, I cannot divide by zero.
