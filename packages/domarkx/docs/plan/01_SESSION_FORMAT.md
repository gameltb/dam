# 01. Domarkx Design: Session Snapshot Format

This document specifies the internal format for a `domarkx` Session Snapshot. The Session Snapshot is an immutable, internal representation of a session, created by the `domarkx import` command. It is designed to be a self-contained, machine-readable representation of the session that can be used to create runnable Session Instances.

The Session Snapshot will be stored as a JSON file.

## Top-Level Structure

The root of the Session Snapshot is a JSON object with the following keys:

-   `version`: (string, required) The version of the session format.
-   `snapshot_id`: (string, required) A unique identifier for the snapshot.
-   `metadata`: (object, required) Metadata about the session.
-   `execution_config`: (object, required) The configuration for the execution environment.
-   `setup_script`: (object, optional) The setup script to be run before the session starts.
-   `conversation`: (array, required) The list of messages in the conversation.

### Example

```json
{
  "version": "1.0",
  "snapshot_id": "uuid-goes-here",
  "metadata": { "...": "..." },
  "execution_config": { "...": "..." },
  "setup_script": { "...": "..." },
  "conversation": [ "...": "..." ]
}
```

---

## `metadata` Object

The `metadata` object contains information about the session snapshot.

-   `source_file`: (string, optional) The path to the original Markdown file that was imported.
-   `import_timestamp`: (string, required) The ISO 8601 timestamp of when the snapshot was created.
-   `user_metadata`: (object, optional) The user-defined metadata from the YAML front matter of the Markdown document.

### Example

```json
"metadata": {
  "source_file": "path/to/original.md",
  "import_timestamp": "2025-11-14T10:45:00Z",
  "user_metadata": {
    "title": "Example Session",
    "author": "Jules"
  }
}
```

---

## `execution_config` Object

The `execution_config` object contains the configuration for the session's execution environment. This is derived from the `session-config` block in the Markdown document.

### Example

```json
"execution_config": {
  "engine": "local_shell",
  "timeout": 60
}
```

---

## `setup_script` Object

The `setup_script` object contains the code that should be executed before the session starts.

-   `language`: (string, required) The programming language of the script.
-   `code`: (string, required) The source code of the script.

### Example

```json
"setup_script": {
  "language": "python",
  "code": "import os\nimport sys\n\nprint('Session initialized.')"
}
```

---

## `conversation` Array

The `conversation` array is a list of `message` objects, representing the conversation between the user and the assistant.

### `message` Object

Each `message` object has the following keys:

-   `role`: (string, required) The role of the speaker (`user` or `assistant`).
-   `message_metadata`: (object, optional) Metadata for the message, from the `msg-metadata` block.
-   `content`: (array, required) A list of `content_block` objects.

### `content_block` Object

A `content_block` object represents a piece of content within a message. It can be of type `text` or `code`.

#### Text Block

-   `type`: (string, required) `"text"`
-   `value`: (string, required) The text content.

#### Code Block

-   `type`: (string, required) `"code"`
-   `language`: (string, required) The language of the code block.
-   `name`: (string, optional) The name of the code block.
-   `code`: (string, required) The source code.

### Example

```json
"conversation": [
  {
    "role": "user",
    "message_metadata": {
      "timestamp": "2023-11-15T14:30:00Z"
    },
    "content": [
      {
        "type": "text",
        "value": "Can you show me the files in the current directory?"
      }
    ]
  },
  {
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "value": "Certainly. Here is the list of files:"
      },
      {
        "type": "code",
        "language": "bash",
        "name": "list_files.sh",
        "code": "ls -la"
      }
    ]
  }
]
```
