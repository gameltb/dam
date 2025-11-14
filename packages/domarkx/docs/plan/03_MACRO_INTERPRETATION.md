# 03. Domarkx Design: Macro Interpretation

This document describes how macros will be handled in the new `domarkx` design. In the new workflow, macros are interpreted once, during the `domarkx import` process. The results of the macro expansion are then stored in the immutable Session Snapshot.

## Macro Interpretation Workflow

1.  **Parsing:** When `domarkx import` is run, the Markdown document is parsed to identify all macro calls (`[@<link_text>](domarkx://<command>...)`).

2.  **Expansion:** For each macro call, the `MacroExpander` class will be invoked to expand the macro. The `MacroExpander` will look up the appropriate macro handler and execute it.

3.  **Replacement:** The macro call in the Markdown document will be replaced with the result of the macro expansion. This could be plain text, a code block, or any other valid Markdown.

4.  **Storage:** The final, expanded Markdown is then used to create the Session Snapshot. The original macro calls are not stored in the snapshot; only the expanded results are.

## Implications of the New Design

This new approach to macro interpretation has several important implications:

*   **Immutability:** Because macros are expanded at import time, the Session Snapshot is fully self-contained and does not require any further macro processing. This makes the snapshots more portable and reproducible.

*   **Simplicity:** The runtime environment for a Session Instance does not need to be aware of the macro system. This simplifies the design of the execution engine.

*   **Static Analysis:** Because the macros are expanded before the session is run, it is possible to perform static analysis on the expanded Markdown to identify potential issues.

*   **Loss of Dynamic Behavior:** The one downside of this approach is that it is no longer possible to have macros that are evaluated at runtime. However, this is a deliberate design choice that prioritizes reproducibility and simplicity over dynamic behavior.

## Example

Consider the following Markdown document:

```markdown
## user

> What is the current date?

## assistant

> The current date is: [@date](domarkx://date)
```

When this document is imported, the `MacroExpander` will be called to expand the `[@date](domarkx://date)` macro. The macro handler will get the current date and return it as a string. The resulting Markdown, which will be used to create the Session Snapshot, will be:

```markdown
## user

> What is the current date?

## assistant

> The current date is: 2025-11-14
```
