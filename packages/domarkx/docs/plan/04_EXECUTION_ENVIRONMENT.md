# 04. Domarkx Design: Execution Environment

This document describes the design of the execution environment for `domarkx`. In the new Docker-like workflow, each Session Instance is provided with its own isolated execution environment, including a dedicated sandbox.

## Execution Workflow

1.  **Instance Creation:** When a user runs `domarkx run <snapshot_id>`, a new Session Instance is created.

2.  **Sandbox Creation:** As part of the instance creation process, a new sandbox is created. The type of sandbox will be determined by the `engine` specified in the `execution_config` of the Session Snapshot (e.g., `local_shell`, `docker`).

3.  **Setup Script Execution:** If a `setup_script` is defined in the Session Snapshot, it is executed in the new sandbox.

4.  **Code Execution:** When the user chooses to execute a code block in the interactive session, the code is sent to the sandbox for execution.

5.  **State Management:** The sandbox maintains the state of the execution environment for the lifetime of the Session Instance. This includes any variables, functions, or files that are created.

6.  **Sandbox Destruction:** When the Session Instance is stopped, the sandbox and all of its associated resources are destroyed.

## Sandboxing and Resource Management

The new design introduces a more robust approach to sandboxing and resource management.

### On-Demand Sandboxes

Sandboxes are created on-demand when a Session Instance is started and are destroyed when the instance is stopped. This has several advantages:

*   **Isolation:** Each Session Instance has its own isolated environment, which prevents interference between sessions.
*   **Reproducibility:** Because each session starts with a clean environment, the results of code execution are more reproducible.
*   **Resource Efficiency:** Resources are only consumed when a session is actively running.

### Pluggable Engines

The execution engine will be designed to be pluggable, allowing for different types of sandboxes to be used. The initial implementation will support the following engines:

*   **`local_shell`:** This engine will execute code in a local shell process. This is the simplest engine, but it is also the least secure.
*   **`docker`:** This engine will execute code in a Docker container. This provides a high level of isolation and security.

In the future, additional engines could be added to support other execution environments, such as remote servers or cloud-based services.

## Tool Executors

The concept of Tool Executors will be retained in the new design, but it will be adapted to the new on-demand sandbox model. When a Session Instance is created, a Tool Executor will be created for each tool that is available in the session. The Tool Executor will be responsible for communicating with the sandbox to execute the tool's code.
