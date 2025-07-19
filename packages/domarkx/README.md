# Domarkx

**Domarkx** (from **Do**cument + **Mark**down + E**x**ecute): Your documentation is not just static text—it's executable, extensible, and powered by LLMs.

---

## Overview

Domarkx transforms your Markdown documentation and LLM chat logs into powerful, interactive sessions. You remain in full control: your Markdown file is the single source of truth, and every action is a command you define. The workflow is flexible and transparent, allowing you to connect with any command-line tool, script, or executable.

## Key Features

- **🛠️ Your Tools, Your Rules:** Define any command you can imagine using a powerful and intuitive placeholder system.
- **📝 Plain Markdown as Source of Truth:** No proprietary formats. Sessions are just Markdown, portable and editable anywhere.
- **🔍 Transparent Actions, Not Magic:** See exactly what commands will be executed before they run. You're always in control.
- ✂️ **Context-Aware Extraction:** Effortlessly refactor and split your sessions with a clear, auditable text history.

---

## Project Structure

```
packages/domarkx/
├── domarkx/
│   ├── __init__.py
│   ├── cli.py               # Main Python CLI entry point
│   ├── action/
│   ├── agents/
│   ├── models/
│   └── tools/
├── docs/
│   ├── developer_guide.md
│   └── design_specification.md
├── tests/
│   └── ...
└── pyproject.toml
```

For more details on the project's design and architecture, please see the [Documentation Format](docs/documentation_format.md). For information on how to contribute to the project, please see the [Developer Guide](docs/developer_guide.md).

---

## Setup Instructions

1.  **Navigate to the `packages/domarkx` directory.**
2.  **Create a virtual environment and activate it:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    uv pip install -e .[all]
    ```

---

## Usage

The primary way to use `domarkx` is through its command-line interface.

**General help:**

```bash
domarkx --help
```

**Execute an entire LLM Markdown session:**

```bash
domarkx exec-doc <your_markdown_file.md>
```

**Execute a specific code block in a conversation:**

```bash
domarkx exec-doc-code-block <your_markdown_file.md> <message_index> <code_block_in_message_index>
```

---

## Contributing

Contributions are welcome! Please see the [Developer Guide](docs/developer_guide.md) for more information.

---

## License

MIT License
