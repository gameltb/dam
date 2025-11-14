# 05. Domarkx Design: Module Design

This document proposes a new module structure for the `domarkx` codebase to support the new Docker-like workflow. The new design will be more modular and extensible, and it will provide a clear separation of concerns between the different components of the system.

## Proposed Module Structure

```
packages/domarkx/src/domarkx/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── import.py
│   ├── snapshots.py
│   ├── run.py
│   ├── instances.py
│   └── export.py
├── data/
│   ├── __init__.py
│   ├── models.py       # Pydantic models for snapshots and instances
│   └── store.py        # For storing and retrieving snapshots and instances
├── session/
│   ├── __init__.py
│   ├── snapshot.py     # Logic for creating and managing snapshots
│   └── instance.py     # Logic for creating and managing instances
├── execution/
│   ├── __init__.py
│   ├── engine.py       # Base class for execution engines
│   ├── local_shell.py  # Local shell execution engine
│   └── docker.py       # Docker execution engine
└── parsing/
    ├── __init__.py
    ├── markdown.py     # For parsing Markdown documents
    └── macros.py       # For expanding macros
```

## Key Classes

### `domarkx.data.models.SessionSnapshot`

A Pydantic model that represents a Session Snapshot. This will be based on the format defined in `01_SESSION_FORMAT.md`.

### `domarkx.data.models.SessionInstance`

A Pydantic model that represents a Session Instance. This will include the state of the instance, such as the conversation history and the status of the execution environment.

### `domarkx.data.store.Store`

A class for storing and retrieving Session Snapshots and Session Instances. The initial implementation will use the local file system for storage.

### `domarkx.session.snapshot.SnapshotManager`

A class for creating and managing Session Snapshots. This will contain the logic for the `domarkx import` command.

### `domarkx.session.instance.InstanceManager`

A class for creating and managing Session Instances. This will contain the logic for the `domarkx run` command.

### `domarkx.execution.engine.Engine`

A base class for execution engines. This will define the interface for creating, running, and destroying sandboxes.

### `domarkx.execution.local_shell.LocalShellEngine`

An implementation of the `Engine` interface that uses a local shell process for execution.

### `domarkx.execution.docker.DockerEngine`

An implementation of the `Engine` interface that uses a Docker container for execution.

### `domarkx.parsing.markdown.MarkdownParser`

A class for parsing Markdown documents and extracting the front matter, session configuration, and conversation.

### `domarkx.parsing.macros.MacroExpander`

A class for expanding macros in a Markdown document. This will be based on the existing `MacroExpander` class, but it will be adapted to the new import-time expansion model.
