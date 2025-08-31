import pathlib
from typing import Annotated

import rich
import rich.markdown
import typer
from rich.console import Console

from domarkx.utils.chat_doc_parser import MarkdownLLMParser


from typing import Any


def exec_doc_code_block(
    doc: Annotated[
        pathlib.Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, writable=True, readable=True, resolve_path=True),
    ],
    message_index: int,
    code_block_in_message_index: int,
) -> None:
    with doc.open() as f:
        md_content = f.read()

    parser = MarkdownLLMParser()
    parsed_doc = parser.parse(md_content)

    if not 0 <= message_index < len(parsed_doc.conversation):
        rich.print(f"[bold red]Error:[/bold red] Message index {message_index} is out of bounds.")
        raise typer.Exit(1)

    message_obj = parsed_doc.conversation[message_index]
    code_blocks = message_obj.code_blocks

    if not 0 <= code_block_in_message_index < len(code_blocks):
        rich.print(
            f"[bold red]Error:[/bold red] Code block index {code_block_in_message_index} is out of bounds for message {message_index}."
        )
        raise typer.Exit(1)

    code_block = code_blocks[code_block_in_message_index]

    console = Console(markup=False)
    if message_obj.content:
        md = rich.markdown.Markdown(message_obj.content)
        console.rule("message")
        console.print(md)
    console.rule("code")
    console.print(rich.markdown.Markdown(f"```{code_block.language}\n{code_block.code}\n```"))
    console.input("Press Enter to exec, Ctrl+C to cancel.")
    console.rule("exec")

    if code_block.language and code_block.language.startswith("python"):
        exec(code_block.code)


def register(main_app: typer.Typer, settings: Any) -> None:
    main_app.command()(exec_doc_code_block)
