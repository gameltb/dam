# Domarkx Markdown Design Conventions

This document outlines the design conventions for creating executable Markdown documents in `domarkx`. Understanding these conventions is crucial for leveraging the full power of the `domarkx` system.

## Core Principles

The design of `domarkx`'s Markdown format is guided by the principle of "executable documentation". The goal is to create documents that are not only human-readable but also machine-parsable and executable. This is achieved by enforcing a structured format that represents a conversation, including user commands and assistant responses with executable code blocks.

## Document Structure

A `domarkx` document is structured as a sequence of messages, optionally preceded by a metadata block (front matter) and a session configuration block.

### 1. Front Matter (Optional)

You can include a YAML front matter block at the beginning of your document to specify global metadata. This block must be enclosed by triple-dashed lines (`---`).

**Example:**

```yaml
---
author: Jane Doe
creation_date: 2023-11-15
tags: [domarkx, tutorial, conventions]
---
```

### 2. Session Configuration (Optional)

You can define a session configuration block to set up the execution environment for the code blocks in the document.

-   The configuration block starts with `` ```session-config `` and ends with ```` ``` ````.
-   The content of this block must be a valid JSON object.
-   Immediately following the configuration block, you can include a code block for session setup tasks, such as importing libraries or defining environment variables.

**Example:**

````markdown
```session-config
{
  "engine": "local_shell",
  "timeout": 60
}
```

```python
import os
import sys

print("Session initialized.")
print(f"Python version: {sys.version}")
```
````

### 3. Conversation Format

The main body of the document is structured as a conversation.

-   Each message begins with a level-2 heading (`## `) followed by the speaker's role (e.g., `## user` or `## assistant`).

### 4. Message Structure

Each message can consist of three parts: metadata, a blockquote for text, and code blocks.

-   **Message Metadata (Optional):** You can include a JSON block with the language specifier `json msg-metadata` to provide metadata for a specific message.

-   **Blockquote:** Use standard Markdown blockquotes (`> `) for the textual content of the message.

-   **Code Blocks:** Use standard Markdown fenced code blocks (``` ```) to specify executable code.
    -   You **must** specify the language of the code (e.g., `python`, `bash`).
    -   You can optionally add a `name` attribute to the code block, which can be used for referencing or saving the code.

**Complete Example:**

```markdown
---
title: "Domarkx Design Conventions Demo"
---

```session-config
{
  "engine": "local_shell"
}
```

## user

```json msg-metadata
{
  "timestamp": "2023-11-15T14:30:00Z"
}
```

> Can you show me the files in the current directory?

## assistant

> Certainly. Here is the list of files:

```bash name=list_files.sh
ls -la
```

## Analysis of the Design

### Advantages

1.  **Structured and Predictable:** The strict format ensures that documents are consistent and easily parsable, making the system reliable.
2.  **Modularity:** The separation of concerns (parsing, execution) allows for easy extension and maintenance. For example, adding support for a new language doesn't require changing the core parser.
3.  **Executable Documentation:** This is the core strength. It's excellent for tutorials, reproducible research, and automating workflows directly within the documentation.
4.  **Conversational Flow:** The format is a natural fit for representing interactions, especially with LLM-based agents.

### Disadvantages

1.  **Rigid Format:** The biggest drawback is the strict adherence to the format. Any deviation can lead to parsing errors. The error handling is not always verbose enough to pinpoint the exact issue.
2.  **Lack of Flexibility:** It cannot process existing Markdown files that do not follow this specific convention, requiring manual conversion.
3.  **Limited Markdown Support:** The parser focuses on its custom structures and may ignore other standard Markdown elements like tables or nested lists within message content.
4.  **Security Concerns:** Executing code from documents introduces potential security risks. While sandboxing can mitigate this, it's a critical aspect to consider, especially with documents from untrusted sources.

## Future Extensions

Based on the current design, here are some potential directions for future development:

1.  **More Flexible Parser:** Improve the parser to be more tolerant of minor formatting variations and provide more detailed error messages.
2.  **Expanded Execution Capabilities:**
    -   Add support for more programming languages (e.g., R, Julia).
    -   Integrate with more execution backends like Docker containers or remote SSH servers.
3.  **Enhanced User Experience:**
    -   Develop an IDE plugin (e.g., for VS Code) with syntax highlighting, autocompletion, and real-time validation.
    -   Support for rendering rich outputs like plots and images directly in the document.
4.  **Improved Security and Management:**
    -   Strengthen the sandboxing environment with fine-grained permission controls.
    -   Integrate dependency management to declare and install necessary packages for code blocks.
```
