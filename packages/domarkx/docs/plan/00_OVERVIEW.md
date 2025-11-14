# 00. Domarkx Design: Overview and Core Concepts

This document provides a high-level overview of the new design for `domarkx`. The new design moves away from a real-time, interactive file model to a more robust, snapshot-based model that is heavily inspired by the Docker container workflow.

## Core Concepts

The new design is centered around three core concepts:

*   **Markdown Document (the "Dockerfile"):** The user-facing Markdown document is the source of truth for creating a session. It is analogous to a `Dockerfile` in that it contains the instructions for building a session snapshot.

*   **Session Snapshot (the "Image"):** A Session Snapshot is an immutable, internal representation of a `domarkx` session. It is created by importing a Markdown document and interpreting all of the macros. This is analogous to a Docker image.

*   **Session Instance (the "Container"):** A Session Instance is a runnable, stateful instance of a Session Snapshot. This is where LLM interactions and code executions happen. Multiple instances can be created from a single snapshot, and each instance has its own isolated state. This is analogous to a Docker container.

## High-Level Workflow

The new workflow is designed to be familiar to users who have experience with Docker.

1.  **`domarkx import <document.md>` (like `docker build`):**
    The user starts by creating a Markdown document with the desired content and macros. They then use the `import` command to create a Session Snapshot. This process involves parsing the Markdown, interpreting the macros, and storing the result in an internal format.

2.  **`domarkx run <session_id>` (like `docker run`):**
    Once a Session Snapshot has been created, the user can use the `run` command to create a Session Instance. This will create a new, isolated environment for the session, including a dedicated sandbox for code execution. The user can then interact with the LLM and execute code within this instance.

3.  **`domarkx export <instance_id> > <new_document.md>` (like `docker export`):**
    At any point, the user can export the current state of a Session Instance to a new Markdown document. This will include any new LLM responses and code execution results. The exported document can then be modified and re-imported to create a new Session Snapshot.

## Analogy to Docker

| `domarkx` Concept     | Docker Analogy | Description                                                                                                     |
| --------------------- | -------------- | --------------------------------------------------------------------------------------------------------------- |
| Markdown Document     | `Dockerfile`   | The source of truth for building a session.                                                                     |
| Session Snapshot      | `Image`        | An immutable, internal representation of a session.                                                             |
| Session Instance      | `Container`    | A runnable, stateful instance of a session.                                                                     |
| `domarkx import`      | `docker build` | Creates a Session Snapshot from a Markdown Document.                                                            |
| `domarkx run`         | `docker run`   | Creates a Session Instance from a Session Snapshot.                                                             |
| `domarkx export`      | `docker export`| Exports the state of a Session Instance to a Markdown Document.                                                 |
