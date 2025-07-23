# Domarkx Documentation Format

## 1. Introduction

This document outlines the design, architecture, and conventions for the `domarkx` system. `domarkx` (from **Do**cument + **Mark**down + E**x**ecute) is a system that turns your Markdown documents and LLM chat logs into executable, interactive sessions. It aims to provide a flexible and transparent workflow for managing and automating tasks using Large Language Models (LLMs).

## 2. Core Concepts

The fundamental principle of `domarkx` is that **your documentation is the source of truth**. Every session, every piece of code, and every interaction is stored in plain Markdown files. This approach ensures that your projects are portable, version-controllable, and easy to understand.

### 2.1. Sessions

A **session** is a single Markdown file that represents a conversation with an LLM. It contains a series of messages, code blocks, and other content that make up the interaction. Each session is a self-contained unit of work that can be executed, modified, and version-controlled using Git.

### 2.2. Templates

**Templates** are pre-defined session files that can be used to create new sessions. They can contain placeholders (macros) that are dynamically replaced with values when a new session is created. This allows for the creation of standardized workflows and boilerplate reduction.

### 2.3. Macros

**Macros** are special commands embedded in Markdown that allow for dynamic content generation and execution of actions. They are the primary mechanism for adding interactivity and automation to your documents.

## 3. Markdown Document Conventions

To ensure consistency and proper parsing, `domarkx` uses a set of conventions for structuring Markdown documents.

### 3.1. Front Matter (Optional)

You can include a YAML front matter block at the beginning of your document to specify global metadata. This block must be enclosed by triple-dashed lines (`---`).

**Example:**

```yaml
---
author: Jane Doe
creation_date: 2023-11-15
tags: [domarkx, tutorial, conventions]
---
```

### 3.2. Session Configuration (Optional)

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

### 3.3. Conversation Format

The main body of the document is structured as a conversation.

-   Each message begins with a level-2 heading (`## `) followed by the speaker's role (e.g., `## user` or `## assistant`).

### 3.4. Message Structure

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

## 4. Macro Implementation and Extension

Macros are the core of `domarkx`'s interactivity. They are parsed from Markdown links and executed by the `MacroExpander` class.

### 4.1. Macro Format

Macros are defined using a specific link format:

`[@<link_text>](domarkx://<command>?<param1>=<value1>&<param2>=<value2>)`
`[@<link_text>](domarkx://<command>)[<param_name>](<param_value>)`

-   `@<link_text>`: The display text of the macro.
-   `<command>`: The name of the command to execute.
-   `<param1>=<value1>&<param2>=<value2>`: The parameters for the command, passed as a query string.
-   `[<param_name>](<param_value>)`: An optional second link immediately following the macro, which is treated as a parameter.

### 4.2. Macro Expansion

The `MacroExpander` class is responsible for finding and expanding macros in a document. It maintains a registry of macro handlers, and when it encounters a macro, it calls the corresponding handler to expand it.

The expansion process is designed to be robust. It performs a single-pass replacement of macros, which avoids issues with nested macros and allows for context-aware expansion in the future.

### 4.3. Extending Macros

To add a new macro, you need to:

1.  Implement a new macro handler function in the `MacroExpander` class.
2.  Register the new macro in the `macros` dictionary in the `MacroExpander`'s constructor.

This modular design makes it easy to extend `domarkx` with new functionality.

## 5. Tool Executors

To provide flexibility in tool execution, `domarkx` introduces the concept of **Tool Executors**. A Tool Executor is a class responsible for running a tool's code in a specific environment. This allows for transparently executing tools locally or in a remote environment, such as a Jupyter kernel running in a Docker container.

### 5.1. Tool Executor Configuration

Tool Executors are configured in the `setup-script` block of a session. This allows users to specify which executor to use for each tool.

```python setup-script
from domarkx.tool_executors.jupyter import JupyterToolExecutor
from domarkx.tools.tool_registry import get_tool
from domarkx.tools.tool_wrapper import ToolWrapper

# Configure the executor
jupyter_executor = JupyterToolExecutor()

# Wrap the clean tool with the executor
read_file_tool = ToolWrapper(tool_func=read_file, executor=jupyter_executor)

tools = [read_file_tool]
```

### 5.2. Clean Tools

To support different execution environments, tool logic is separated from the `domarkx` framework. These **Portable Tools** are simple Python functions that do not have any `domarkx`-specific dependencies or decorators. They are located in the `packages/domarkx/domarkx/tools/portable_tools` subpackage.

### 5.3. Tool Wrappers

A **Tool Wrapper** is a class that inherits from `autogen_core.tools.BaseTool` and makes a clean tool available to the `domarkx` system. The wrapper is responsible for invoking the appropriate Tool Executor to run the tool's logic. This design ensures that the tools remain compatible with the `autogen` ecosystem.

## 6. Relationship Between Macros and Tools

In `domarkx`, **Macros** and **Tools** are distinct but related concepts that work together to provide interactivity and extensibility.

-   **Macros**: A macro is a user-facing command embedded in a Markdown document using a special link format (`domarkx://...`). Macros are the primary way for users to trigger actions directly from the documentation. They are discovered and parsed by the `MacroExpander`.

-   **Tools**: A tool is a Python function that encapsulates a specific piece of logic (e.g., reading a file, running a command). Tools are designed to be executed by **Tool Executors**, which allows them to run in different environments (e.g., locally or in a Jupyter kernel).

The relationship between them is as follows: a **Macro**'s primary role is often to invoke a **Tool**. The macro handler function, upon parsing the `domarkx://` link, will typically call the relevant tool, passing along any parameters from the macro. This separation of concerns allows for a clean architecture where user interaction (Macros) is decoupled from the underlying implementation (Tools).

## 7. Future Development

-   **Context-Aware Macro Expansion**: Enhance the macro expansion process to provide more context to macro handlers, such as the current file path and line number.
-   **Improved Error Handling**: Provide more detailed and user-friendly error messages for parsing and execution errors.
-   **More Execution Engines**: Add support for more code execution engines, such as remote Docker containers.
-   **GUI**: Develop a graphical user interface for a more user-friendly experience.
