# Domarkx Design Specification

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

### 3.1. Frontmatter

Every `domarkx` document should start with a YAML frontmatter block that contains metadata about the document.

```yaml
---
title: "Session Title"
version: "0.0.1"
---
```

### 3.2. Session Configuration


At the beginning of the document and before each message, any number of code blocks are allowed. Code blocks can be of any language (e.g., `python setup-script`, `json session-config`, etc.), and `session-config` is just a special type of code block.

Example:
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

client = OpenRouterR1OpenAIChatCompletionClient(...)
tools = [...]
```

### 3.3. Setup Script

The setup script is a Python code block with the `setup-script` language identifier. This script is executed before the session starts and is used to configure the environment, including the `model_client` and `tools` that the agent will use.

```python setup-script
from domarkx.models.openrouter import OpenRouterR1OpenAIChatCompletionClient
import os

client = OpenRouterR1OpenAIChatCompletionClient(...)
tools = [...]
```

### 3.4. System Message


Each message may contain at most one blockquote, which serves as the content for system, user, or assistant messages. The blockquote can span multiple consecutive lines, and only one blockquote is allowed per message.

Example:
```
> You are a helpful assistant.
> This is a multi-line blockquote.
> It will be parsed as one blockquote content.
```

### 3.5. User/Assistant Messages


Each message starts with `## User` or `## Assistant`, followed by any number of code blocks (such as metadata blocks, script blocks, etc.), and at most one blockquote (which can span multiple lines). Each message must contain at least one code block or one blockquote, or both.

Example:
```markdown
## User

```json msg-metadata
{
  "source": "user",
  "type": "UserMessage"
}
```

```python
print("hello")
```

> User's message content.
> This is a multi-line blockquote.
```

## 4. Macro Implementation and Extension

Macros are the core of `domarkx`'s interactivity. They are parsed from Markdown links and executed by the `MacroExpander` class.

### 4.1. Macro Format

Macros are defined using a specific link format:

`[@<link_text>](domarkx://<command>?<param1>=<value1>&<param2>=<value2>)`

-   `@<link_text>`: The display text of the macro.
-   `<command>`: The name of the command to execute.
-   `<param1>=<value1>&<param2>=<value2>`: The parameters for the command.

### 4.2. Macro Expansion

The `MacroExpander` class is responsible for finding and expanding macros in a document. It maintains a registry of macro handlers, and when it encounters a macro, it calls the corresponding handler to expand it.

The expansion process is designed to be robust. It performs a single-pass replacement of macros, which avoids issues with nested macros and allows for context-aware expansion in the future.

### 4.3. Extending Macros

To add a new macro, you need to:

1.  Implement a new macro handler function in the `MacroExpander` class.
2.  Register the new macro in the `macros` dictionary in the `MacroExpander`'s constructor.

This modular design makes it easy to extend `domarkx` with new functionality.

## 5. Future Development

-   **Context-Aware Macro Expansion**: Enhance the macro expansion process to provide more context to macro handlers, such as the current file path and line number.
-   **Improved Error Handling**: Provide more detailed and user-friendly error messages for parsing and execution errors.
-   **More Execution Engines**: Add support for more code execution engines, such as remote Docker containers.
-   **GUI**: Develop a graphical user interface for a more user-friendly experience.

## 6. Tool Executors

To provide flexibility in tool execution, `domarkx` introduces the concept of **Tool Executors**. A Tool Executor is a class responsible for running a tool's code in a specific environment. This allows for transparently executing tools locally or in a remote environment, such as a Jupyter kernel running in a Docker container.

### 6.1. Tool Executor Configuration

Tool Executors are configured in the `setup-script` block of a session. This allows users to specify which executor to use for each tool.

```python setup-script
from domarkx.tool_executors.jupyter import JupyterToolExecutor
from domarkx.tools.unconstrained.read_file import read_file
from domarkx.tools.tool_wrapper import ToolWrapper

# Configure the executor
jupyter_executor = JupyterToolExecutor()

# Wrap the clean tool with the executor
read_file_tool = ToolWrapper(tool_func=read_file, executor=jupyter_executor)

tools = [read_file_tool]
```

### 6.2. Clean Tools

To support different execution environments, tool logic is separated from the `domarkx` framework. These **Clean Tools** are simple Python functions that do not have any `domarkx`-specific dependencies or decorators. They are located in the `packages/domarkx/domarkx/tools/unconstrained` subpackage.

### 6.3. Tool Wrappers

A **Tool Wrapper** is a class that inherits from `autogen_core.tools.BaseTool` and makes a clean tool available to the `domarkx` system. The wrapper is responsible for invoking the appropriate Tool Executor to run the tool's logic. This design ensures that the tools remain compatible with the `autogen` ecosystem.
