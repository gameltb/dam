# AI Agents Documentation

This document describes the AI agents available within the Gemini CLI environment to assist with development tasks.

## 1. Gemini CLI Agent (Primary)

The primary agent is an interactive CLI assistant specializing in software engineering tasks.

### Capabilities

- **File System Operations**: Read, write, list, search, and modify files.
- **Code Execution**: Run shell commands, build projects, and execute tests.
- **Code Modification**: Perform precise, surgical text replacements in code files.
- **Web Interaction**: Fetch web pages, search Google, and perform browser automation (Puppeteer-like).
- **Planning**: Break down complex tasks into todo lists and execute them step-by-step.

### Role

- Acting as a pair programmer.
- Refactoring code.
- Fixing bugs.
- Setting up new projects or features.
- Answering questions about the codebase.

## 2. Codebase Investigator (Sub-Agent)

A specialized tool/agent invoked by the primary agent for deep analysis.

### Capabilities

- **Architectural Mapping**: Understanding the high-level structure of the system.
- **Dependency Analysis**: Tracing how modules and components interact.
- **Root Cause Analysis**: Investigating complex bugs that span multiple files.
- **Vague Request Handling**: clarifying intent when the user asks broad questions like "How does the auth system work?".

### Usage

The primary agent delegates to the `codebase_investigator` when:

- The request requires reading/indexing a large number of files.
- The user asks a high-level question about the system architecture.
- A bug's location is unknown and requires "detective work."

## Workflow

1.  **User Request**: User inputs a command or question.
2.  **Primary Agent**: Analyzes the request.
    - If simple: Solves it directly using standard tools.
    - If complex/vague: Delegates to `codebase_investigator`.
3.  **Investigation**: Sub-agent analyzes the codebase and returns a report.
4.  **Action**: Primary agent uses the insights to formulate a plan and execute changes.
