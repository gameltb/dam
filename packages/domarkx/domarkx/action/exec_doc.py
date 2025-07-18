import asyncio
import copy
import pathlib
from typing import Annotated

import typer
from prompt_toolkit import PromptSession
from rich.console import Console

import domarkx.ui.console
from domarkx.autogen_session import AutoGenSession
from domarkx.utils.chat_doc_parser import Message, append_message


async def aexec_doc(doc: pathlib.Path, handle_one_toolcall: bool = False):
    console = Console(markup=False)
    session = AutoGenSession(doc)
    await session.setup()

    console.print("".join(session.doc.raw_lines))

    # console.input("Press Enter to run stream, Ctrl+C to cancel.")

    while True:
        task_msg = None
        latest_msg = session.messages[-1] if len(session.messages) > 0 else None
        if len(session.messages) == 0 or (
            latest_msg.get("type", "") not in ["UserMessage"] and "content" in latest_msg
        ):
            task_msg: str = await PromptSession().prompt_async(
                "task > ",
                multiline=True,
                bottom_toolbar="press Alt+Enter in order to accept the input. (Or Escape followed by Enter.)",
            )
            if latest_msg and latest_msg.get("type", "") in ["FunctionExecutionResultMessage"]:
                if len(task_msg.strip()) == 0:
                    task_msg = None

        response = await domarkx.ui.console.Console(
            session.agent.run_stream(task=task_msg), output_stats=True, exit_after_one_toolcall=handle_one_toolcall
        )

        new_state = await session.agent.save_state()

        _append_new_messages(doc, new_state, session.messages)

        session.messages = new_state["llm_context"]["messages"]

        if handle_one_toolcall:
            break
        user_input: str = await PromptSession().prompt_async("input r to continue > ")
        user_input = user_input.strip().lower()
        if len(user_input) != 0 and user_input != "r":
            break


def _append_new_messages(doc, new_state, messages):
    for message in new_state["llm_context"]["messages"][len(messages) :]:
        message: dict = copy.deepcopy(message)
        content = ""
        if "content" in message:
            if isinstance(message["content"], str):
                content = message.pop("content")
            elif isinstance(message["content"], list) and len(message["content"]) == 1:
                content = message["content"][0].pop("content", "")
        thought = message.pop("thought", "")
        if thought:
            thought = "\n".join("> " + line for line in f"""<think>{thought}</think>""".splitlines())
            content = f"""
{thought}

{content}"""
        with doc.open("a") as f:
            append_message(f, Message("assistant", content, message))


def exec_doc(
    doc: Annotated[
        pathlib.Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, writable=True, readable=True, resolve_path=True),
    ],
    handle_one_toolcall: bool = False,
):
    asyncio.run(aexec_doc(doc, handle_one_toolcall))


def register(main_app: typer.Typer, settings):
    main_app.command()(exec_doc)
