import asyncio
import pathlib
import tempfile
from datetime import datetime
from typing import Annotated

import typer
from domarkx.config import settings
from domarkx.macro_expander import MacroExpander
from prompt_toolkit import PromptSession
from rich.console import Console

import domarkx.ui.console
from domarkx.autogen_session import AutoGenSession
from domarkx.ui.console import PROMPT_TOOLKIT_IS_MULTILINE_CONDITION


async def aexec_doc(
    doc: pathlib.Path,
    handle_one_toolcall: bool = False,
    allow_user_message_in_FunctionExecution=True,
    overwrite: bool = False,
):
    console = Console(markup=False)

    # Read the content from the document
    with open(doc, "r") as f:
        content = f.read()

    # Expand macros
    expander = MacroExpander(base_dir=str(doc.parent))
    expanded_content = expander.expand(content)

    sessions_dir = pathlib.Path(settings.project_path) / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    doc_to_exec = doc
    project_path = pathlib.Path(settings.project_path)
    # Check if the file is under the project path
    if project_path in doc.parents and sessions_dir not in doc.parents:
        # Create a temporary file in the sessions folder
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        temp_filename = f"{now}_{doc.stem}.md"
        doc_to_exec = sessions_dir / temp_filename

        with open(doc_to_exec, "w") as f:
            f.write(expanded_content)
    else:
        if not overwrite:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_filename = f"{doc.stem}_{now}.md"
            doc_to_exec = doc.with_name(new_filename)

        with open(doc_to_exec, "w") as f:
            f.write(expanded_content)


    session = AutoGenSession(doc_to_exec)
    await session.setup()

    while True:
        task_msg = None
        latest_msg = session.messages[-1] if len(session.messages) > 0 else None
        if (
            not allow_user_message_in_FunctionExecution
            and latest_msg
            and isinstance(latest_msg.get("content", ""), list)
        ):
            # If the last message is a FunctionExecutionMessage, we don't want to prompt for user input.
            task_msg = None
        elif len(session.messages) == 0 or (
            latest_msg.get("type", "") not in ["UserMessage"] and "content" in latest_msg
        ):
            task_msg: str = await PromptSession().prompt_async(
                "task > ",
                multiline=PROMPT_TOOLKIT_IS_MULTILINE_CONDITION,
                bottom_toolbar=lambda: "press Alt+Enter in order to accept the input. (Or Escape followed by Enter.)"
                if PROMPT_TOOLKIT_IS_MULTILINE_CONDITION()
                else None,
            )
            if latest_msg and latest_msg.get("type", "") in ["FunctionExecutionResultMessage"]:
                if len(task_msg.strip()) == 0:
                    task_msg = None

        response = await domarkx.ui.console.Console(
            session.agent.run_stream(task=task_msg), output_stats=True, exit_after_one_toolcall=handle_one_toolcall
        )

        new_state = await session.agent.save_state()

        session.append_new_messages(new_state)

        session.messages = new_state["llm_context"]["messages"]

        if handle_one_toolcall:
            break
        user_input: str = await PromptSession().prompt_async("input r to continue > ")
        user_input = user_input.strip().lower()
        if len(user_input) != 0 and user_input != "r":
            break


def exec_doc(
    doc: Annotated[
        pathlib.Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, writable=True, readable=True, resolve_path=True),
    ],
    handle_one_toolcall: bool = False,
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite the original file in the sessions folder."),
):
    asyncio.run(aexec_doc(doc, handle_one_toolcall, overwrite=overwrite))


def register(main_app: typer.Typer, settings):
    main_app.command()(exec_doc)
