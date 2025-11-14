# 02. Domarkx Design: User Workflow and CLI

This document details the user workflow and the command-line interface (CLI) for the new Docker-like `domarkx` system.

## User Workflow

The user workflow is designed to be simple and intuitive, especially for users who are familiar with Docker.

1.  **Author a Markdown Document:** The user begins by creating a standard Markdown document. They can include front matter, session configuration, and a conversation with code blocks. They can also use macros, which will be interpreted during the import process.

2.  **Import the Document:** The user imports the document into `domarkx` using the `domarkx import` command. This creates an immutable Session Snapshot.

3.  **Run the Session:** The user can then create a runnable Session Instance from the snapshot using the `domarkx run` command. This will start an interactive session where they can execute code blocks and get responses from the LLM.

4.  **Export the Session:** At any point, the user can export the current state of a Session Instance to a new Markdown document using the `domarkx export` command.

5.  **Iterate:** The user can then modify the exported Markdown document and re-import it to create a new Session Snapshot, allowing for an iterative workflow.

## Command-Line Interface

The `domarkx` CLI will be updated to support the new workflow.

---

### `domarkx import`

**Usage:** `domarkx import <markdown_file>`

**Description:** Creates a new Session Snapshot from a Markdown file.

**Process:**
1.  Parses the Markdown file.
2.  Interprets any macros in the document.
3.  Creates a new Session Snapshot in the internal format.
4.  Stores the snapshot in the `domarkx` data directory.
5.  Prints the new `snapshot_id` to the console.

---

### `domarkx snapshots`

**Usage:** `domarkx snapshots`

**Description:** Lists all available Session Snapshots.

**Output:** A table with the following columns:
-   `SNAPSHOT ID`
-   `SOURCE FILE`
-   `CREATED`

---

### `domarkx run`

**Usage:** `domarkx run <snapshot_id>`

**Description:** Creates and runs a new Session Instance from a Session Snapshot.

**Process:**
1.  Loads the specified Session Snapshot.
2.  Creates a new Session Instance with its own isolated state.
3.  Creates a new sandbox for code execution.
4.  Starts an interactive session, allowing the user to:
    -   Execute code blocks.
    -   Send messages to the LLM.
    -   Receive responses from the LLM.
5.  When the session is stopped, the sandbox is destroyed.

---

### `domarkx instances`

**Usage:** `domarkx instances`

**Description:** Lists all running Session Instances.

**Output:** A table with the following columns:
-   `INSTANCE ID`
-   `SNAPSHOT ID`
-   `STATUS`

---

### `domarkx export`

**Usage:** `domarkx export <instance_id>`

**Description:** Exports the current state of a Session Instance to a Markdown document.

**Process:**
1.  Loads the specified Session Instance.
2.  Converts the current state of the instance (including new messages and code outputs) into a Markdown document.
3.  Prints the Markdown to `stdout`.

**Example:**
`domarkx export <instance_id> > new_document.md`
